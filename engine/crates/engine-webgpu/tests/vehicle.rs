//! Vehicle controller envelopes on the native wgpu host (same WGSL as Chrome).
//!
//! Analytic checks: static load balance, settling, straight-line drive, braking,
//! steering direction, parking without creep, severing parts and wheels.

use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_vehicle::{
    presets, FunctionalState, VehicleDesc, VehicleHandle, VehicleInput, WheelDesc,
};
use engine_webgpu::WgpuSim;

const DT: f32 = 1.0 / 60.0;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32_768,
        max_joints: 64,
        max_candidates: 65_536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn static_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3], rot: [f32; 4]) -> u32 {
    let body = sim.add_body(BodyState {
        position: pos,
        inv_mass: 0.0,
        orientation: rot,
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: rot,
    });
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [
            half[0],
            half[1],
            half[2],
            (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
        ],
    });
    body
}

/// Flat ground with its top face at y = 0.
fn world() -> WgpuSim {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    // Dry asphalt for the ground (material 0); tyres combine with it Jolt-style.
    sim.set_material(
        0,
        Material {
            friction: 1.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    static_box(
        &mut sim,
        [0.0, -5.0, 0.0],
        [400.0, 5.0, 400.0],
        [0.0, 0.0, 0.0, 1.0],
    );
    sim
}

fn spawn(sim: &mut WgpuSim, desc: &VehicleDesc, y: f32) -> VehicleHandle {
    sim.add_vehicle(desc, [0.0, y, 0.0], [0.0, 0.0, 0.0, 1.0])
        .expect("vehicle")
}

fn run(sim: &mut WgpuSim, steps: u32) {
    for _ in 0..steps {
        sim.step().expect("step");
    }
}

fn chassis(sim: &WgpuSim, v: VehicleHandle) -> BodyState {
    sim.read_body(sim.vehicle_host().chassis_body(v))
}

fn mass_of(desc: &VehicleDesc) -> f32 {
    desc.parts.iter().map(|p| p.mass).sum::<f32>() + desc.wheels.iter().map(|w| w.mass).sum::<f32>()
}

fn speed(b: &BodyState) -> f32 {
    (b.linear_vel[0].powi(2) + b.linear_vel[1].powi(2) + b.linear_vel[2].powi(2)).sqrt()
}

#[test]
fn sedan_settles_on_four_wheels_with_weight_on_the_suspension() {
    let mut sim = world();
    let desc = presets::sedan();
    let v = spawn(&mut sim, &desc, 1.0);
    run(&mut sim, 240);
    let st = sim.vehicle_state(v);
    let c = chassis(&sim, v);
    eprintln!(
        "chassis y {:.4} v {:?} w {:?}",
        c.position[1], c.linear_vel, c.angular_vel
    );
    for (i, w) in st.wheels.iter().enumerate() {
        eprintln!(
            "wheel {i}: contact {} L {:.3} Fs {:.1} pos {:?} omega {:.3}",
            w.in_contact, w.suspension_length, w.suspension_force, w.position, w.omega
        );
    }
    assert_eq!(st.contacts, 4, "all four wheels on the ground");
    let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    let weight = mass_of(&desc) * 10.0;
    assert!(
        (total - weight).abs() < 0.03 * weight,
        "suspension carries the weight: {total} vs {weight}"
    );
    assert!(speed(&c) < 0.05, "chassis at rest: {:?}", c.linear_vel);
    for w in &st.wheels {
        assert!(
            w.suspension_length > 0.15 && w.suspension_length < 0.45,
            "travel within limits: {}",
            w.suspension_length
        );
        // Wheel bottoms touch the ground.
        assert!(
            (w.position[1] - 0.34).abs() < 0.02,
            "wheel centre at radius height: {}",
            w.position[1]
        );
    }
}

fn drive(sim: &mut WgpuSim, v: VehicleHandle, input: VehicleInput, steps: u32) {
    sim.set_vehicle_input(v, input);
    run(sim, steps);
}

fn throttle(forward: f32, steer: f32) -> VehicleInput {
    VehicleInput {
        forward,
        steer,
        ..Default::default()
    }
}

#[test]
fn sedan_accelerates_straight_and_shifts_up() {
    let mut sim = world();
    let v = spawn(&mut sim, &presets::sedan(), 1.0);
    run(&mut sim, 60);
    let mut max_gear = 0;
    for _ in 0..8 {
        drive(&mut sim, v, throttle(1.0, 0.0), 45);
        let st = sim.vehicle_state(v);
        max_gear = max_gear.max(st.gear);
        let c = chassis(&sim, v);
        eprintln!(
            "speed {:.2} rpm {:.0} gear {} clutch {:.2} pos {:?} slip {:?}",
            st.forward_speed,
            st.engine_rpm,
            st.gear,
            st.clutch,
            c.position,
            st.wheels.iter().map(|w| w.slip_ratio).collect::<Vec<_>>()
        );
        eprintln!(
            "   Fn {:?} fL {:?} omega {:?}",
            st.wheels
                .iter()
                .map(|w| w.suspension_force as i32)
                .collect::<Vec<_>>(),
            st.wheels
                .iter()
                .map(|w| w.longitudinal_force as i32)
                .collect::<Vec<_>>(),
            st.wheels.iter().map(|w| w.omega).collect::<Vec<_>>()
        );
    }
    let st = sim.vehicle_state(v);
    let c = chassis(&sim, v);
    assert!(
        st.forward_speed > 15.0,
        "reaches highway speed in 6 s: {}",
        st.forward_speed
    );
    assert!(max_gear >= 3, "automatic box shifts up: {max_gear}");
    assert!(st.engine_rpm < 6801.0);
    assert!(
        c.position[0].abs() < 0.5,
        "drives straight: x drift {}",
        c.position[0]
    );
    assert!(c.position[2] > 30.0, "moved forward: {}", c.position[2]);
}

#[test]
fn steering_right_turns_right_and_brakes_stop_the_car() {
    let mut sim = world();
    let v = spawn(&mut sim, &presets::sedan(), 1.0);
    run(&mut sim, 60);
    drive(&mut sim, v, throttle(0.6, 0.0), 120);
    drive(&mut sim, v, throttle(0.3, 1.0), 90);
    let st = sim.vehicle_state(v);
    let c = chassis(&sim, v);
    eprintln!(
        "turning: yaw {:.3} pos {:?} speed {:.2}",
        st.yaw_rate, c.position, st.forward_speed
    );
    assert!(
        st.yaw_rate < -0.1,
        "right steer yaws right (negative about +Y): {}",
        st.yaw_rate
    );
    assert!(
        c.position[0] < -0.5,
        "heads to the right (-X): {}",
        c.position[0]
    );
    // Brake to a stop.
    drive(
        &mut sim,
        v,
        VehicleInput {
            brake: 1.0,
            ..Default::default()
        },
        240,
    );
    let st = sim.vehicle_state(v);
    eprintln!("braked: speed {:.3}", st.forward_speed);
    assert!(
        st.forward_speed.abs() < 0.05,
        "stopped: {}",
        st.forward_speed
    );
}

#[test]
fn handbrake_holds_on_a_slope_without_creep() {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    // 12° slope rising toward +Z (rotation about +X by -12°).
    let a = -12f32.to_radians();
    let rot = [(a * 0.5).sin(), 0.0, 0.0, (a * 0.5).cos()];
    static_box(&mut sim, [0.0, -5.0, 0.0], [60.0, 5.0, 60.0], rot);
    let v = sim
        .add_vehicle(&presets::sedan(), [0.0, 1.2, 0.0], rot)
        .expect("vehicle");
    sim.set_vehicle_input(
        v,
        VehicleInput {
            handbrake: 1.0,
            brake: 1.0,
            ..Default::default()
        },
    );
    run(&mut sim, 120);
    let start = chassis(&sim, v).position;
    run(&mut sim, 300);
    let end = chassis(&sim, v);
    let drift =
        ((end.position[0] - start[0]).powi(2) + (end.position[2] - start[2]).powi(2)).sqrt();
    eprintln!(
        "slope drift {:.5} m over 5 s; sleeping {}",
        drift,
        end.is_sleeping()
    );
    assert!(drift < 0.005, "parked car creeps {drift} m");
}

#[test]
fn detached_wheel_keeps_its_spin_and_the_car_loses_its_mass() {
    let mut sim = world();
    let desc = presets::sedan();
    let v = spawn(&mut sim, &desc, 1.0);
    run(&mut sim, 60);
    drive(&mut sim, v, throttle(1.0, 0.0), 120);
    let before = sim.vehicle_state(v);
    let omega = before.wheels[0].omega;
    let c0 = chassis(&sim, v);
    sim.detach_wheel(v, 0);
    sim.step().expect("step");
    let wb = sim.vehicle_host().wheel_body(v, 0);
    let wheel = sim.read_body(wb);
    let c1 = chassis(&sim, v);
    let st = sim.vehicle_state(v);
    eprintln!(
        "omega {omega:.2} wheel body w {:?} v {:?} flags {}",
        wheel.angular_vel, wheel.linear_vel, wheel.flags
    );
    assert!(wheel.is_dynamic(), "wheel became a rigid body");
    let spin = wheel.angular_vel[0];
    assert!(
        (spin - omega).abs() < 0.15 * omega.abs() + 0.5,
        "spin carried over: {spin} vs {omega}"
    );
    assert!(speed(&wheel) > 0.5 * speed(&c0), "wheel moves with the car");
    assert_eq!(st.wheel_mask & 1, 0, "wheel 0 removed from the vehicle");
    let m0 = 1.0 / c0.inv_mass;
    let m1 = 1.0 / c1.inv_mass;
    assert!(
        (m0 - m1 - desc.wheels[0].mass).abs() < 0.05,
        "chassis mass drops by the wheel: {m0} -> {m1}"
    );
    // The car keeps driving on three wheels without blowing up.
    run(&mut sim, 120);
    let c = chassis(&sim, v);
    assert!(c.position.iter().all(|x| x.is_finite()) && speed(&c) < 60.0);
}

#[test]
fn severing_the_engine_part_cuts_drive() {
    let mut sim = world();
    let desc = presets::destructible_sedan();
    let v = spawn(&mut sim, &desc, 1.0);
    run(&mut sim, 60);
    let engine = desc.parts.iter().position(|p| p.name == "engine").unwrap() as u32;
    sim.detach_part(v, engine);
    sim.step().expect("step");
    let st = sim.vehicle_state(v);
    assert!(
        !st.engine_connected,
        "engine part gone → engine disconnected"
    );
    assert!(st.parts_lost & (1 << engine) != 0);
    let part_body = sim.read_body(sim.vehicle_host().part_body(v, engine));
    assert!(part_body.is_dynamic(), "engine block is a free rigid body");
    drive(&mut sim, v, throttle(1.0, 0.0), 120);
    let st = sim.vehicle_state(v);
    eprintln!("no-engine speed {:.3}", st.forward_speed);
    assert!(
        st.forward_speed.abs() < 0.3,
        "no drive without an engine: {}",
        st.forward_speed
    );
}

#[test]
fn losing_the_driveshaft_cuts_drive_even_with_overrides() {
    let mut sim = world();
    let desc = presets::destructible_sedan();
    let v = spawn(&mut sim, &desc, 1.0);
    run(&mut sim, 60);
    let shaft = desc
        .parts
        .iter()
        .position(|p| p.name == "driveshaft")
        .unwrap() as u32;
    sim.disable_part(v, shaft);
    drive(&mut sim, v, throttle(1.0, 0.0), 120);
    let st = sim.vehicle_state(v);
    assert_eq!(st.drive_mask, 0, "rear wheels lost drive");
    assert!(
        st.forward_speed.abs() < 0.3,
        "no drive: {}",
        st.forward_speed
    );
    // Overrides are ANDed with ownership: they cannot resurrect the shaft.
    sim.set_functional_state(v, FunctionalState::default());
    run(&mut sim, 2);
    assert_eq!(sim.vehicle_state(v).drive_mask, 0);
}

#[test]
fn spare_slot_wheel_can_be_added_at_runtime() {
    let mut sim = world();
    let mut desc = presets::sedan();
    desc.spare_wheels = 2;
    let v = spawn(&mut sim, &desc, 1.0);
    run(&mut sim, 120);
    let m0 = 1.0 / chassis(&sim, v).inv_mass;
    let extra = WheelDesc {
        position: [0.82, -0.05, 0.0],
        ..desc.wheels[2].clone()
    };
    let idx = sim.add_wheel(v, &extra).expect("spare slot");
    assert_eq!(idx, 4);
    run(&mut sim, 120);
    let st = sim.vehicle_state(v);
    let m1 = 1.0 / chassis(&sim, v).inv_mass;
    assert_eq!(st.wheels.len(), 5);
    assert!(st.wheels[4].in_contact, "new wheel reaches the ground");
    assert!(
        (m1 - m0 - extra.mass).abs() < 0.05,
        "mass added: {m0} -> {m1}"
    );
    let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    let weight = m1 * 10.0;
    assert!(
        (total - weight).abs() < 0.05 * weight,
        "five wheels carry the weight: {total} vs {weight}"
    );
}

#[test]
fn parked_vehicle_sleeps_and_input_wakes_it() {
    let mut sim = world();
    let v = spawn(&mut sim, &presets::sedan(), 1.0);
    run(&mut sim, 240);
    assert!(chassis(&sim, v).is_sleeping(), "parked car sleeps");
    drive(&mut sim, v, throttle(1.0, 0.0), 30);
    let c = chassis(&sim, v);
    assert!(!c.is_sleeping(), "throttle wakes it");
    assert!(sim.vehicle_state(v).forward_speed > 0.2);
}

#[test]
fn snapshot_restore_replays_the_same_drive_and_events_report_severing() {
    let mut sim = world();
    let v = spawn(&mut sim, &presets::destructible_sedan(), 1.0);
    run(&mut sim, 60);
    let _ = sim.poll_vehicle_events();
    let snap = sim.save_state();
    let chassis_body = sim.vehicle_host().chassis_body(v);
    let mut first = Vec::new();
    let mut events = Vec::new();
    for k in 0..3 {
        drive(
            &mut sim,
            v,
            throttle(1.0, if k == 1 { 0.5 } else { 0.0 }),
            40,
        );
        if k == 1 {
            sim.detach_wheel(v, 2);
            sim.disable_part(v, 3);
        }
        events.extend(sim.poll_vehicle_events());
        first.push(sim.read_body(chassis_body));
    }
    assert!(
        events.iter().any(|e| matches!(
            e,
            engine_vehicle::VehicleEvent::WheelDetached { wheel: 2, .. }
        )),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, engine_vehicle::VehicleEvent::PartLost { part: 3, .. })),
        "{events:?}"
    );
    assert!(events.iter().any(|e| matches!(
        e,
        engine_vehicle::VehicleEvent::FunctionalStateChanged {
            engine_connected: false,
            ..
        }
    )));

    sim.restore_state(&snap);
    let mut second = Vec::new();
    for k in 0..3 {
        drive(
            &mut sim,
            v,
            throttle(1.0, if k == 1 { 0.5 } else { 0.0 }),
            40,
        );
        if k == 1 {
            sim.detach_wheel(v, 2);
            sim.disable_part(v, 3);
        }
        second.push(sim.read_body(chassis_body));
    }
    for (a, b) in first.iter().zip(&second) {
        let d: f32 = (0..3).map(|i| (a.position[i] - b.position[i]).abs()).sum();
        eprintln!("replay Δ {d:e}: {:?} vs {:?}", a.position, b.position);
        assert!(d < 1e-3, "replay diverges by {d}");
    }
}

#[test]
fn sphere_cast_wheels_climb_a_kerb() {
    let mut sim = world();
    // A 15 cm kerb (step up, plateau, step down) across the road, 8 m ahead.
    static_box(
        &mut sim,
        [0.0, 0.075, 10.0],
        [6.0, 0.075, 1.5],
        [0.0, 0.0, 0.0, 1.0],
    );
    let mut desc = presets::sedan();
    desc.wheel_query = engine_vehicle::WheelQuery::Sphere;
    let v = spawn(&mut sim, &desc, 1.0);
    run(&mut sim, 60);
    let settled = chassis(&sim, v).position[1];
    let mut peak = settled;
    sim.set_vehicle_input(v, throttle(0.5, 0.0));
    for _ in 0..240 {
        sim.step().expect("step");
        let c = chassis(&sim, v);
        peak = peak.max(c.position[1]);
        if c.position[2] > 9.0 && c.position[2] < 11.0 {
            let st = sim.vehicle_state(v);
            eprintln!(
                "on kerb: z {:.2} y {:.3} contacts {} L {:?}",
                c.position[2],
                c.position[1],
                st.contacts,
                st.wheels
                    .iter()
                    .map(|w| (w.suspension_length * 100.0).round() / 100.0)
                    .collect::<Vec<_>>()
            );
        }
    }
    let c = chassis(&sim, v);
    eprintln!(
        "kerb: settled {settled:.3} peak {peak:.3} end {:?}",
        c.position
    );
    assert!(
        c.position[2] > 14.0,
        "drove over the kerb: z {}",
        c.position[2]
    );
    assert!(
        peak > settled + 0.08 && peak < settled + 0.5,
        "rode up onto the kerb: {settled} → {peak}"
    );
    let q = glam::Quat::from_array(c.orientation);
    assert!((q * glam::Vec3::Y).y > 0.95, "level after the kerb");
}

fn obstacle_course(sim: &mut WgpuSim) {
    static_box(
        sim,
        [0.0, 0.075, 12.0],
        [6.0, 0.075, 1.5],
        [0.0, 0.0, 0.0, 1.0],
    );
    let a = -0.15f32;
    static_box(
        sim,
        [3.0, 0.4, 25.0],
        [3.0, 0.3, 5.0],
        [(a * 0.5).sin(), 0.0, 0.0, (a * 0.5).cos()],
    );
    for i in 0..6 {
        let body = sim.add_body(BodyState::dynamic(
            glam::Vec3::new(-2.0 + i as f32 * 0.9, 0.3, 35.0),
            glam::Quat::IDENTITY,
            30.0,
            glam::Vec3::splat(30.0 * 0.6 * 0.6 / 6.0),
        ));
        sim.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.3, 0.3, 0.3, 0.52],
        });
    }
}

fn course_run(full_scan: bool, preset: VehicleDesc) -> Vec<[f32; 3]> {
    let mut sim = world();
    obstacle_course(&mut sim);
    sim.set_vehicle_full_scan(full_scan);
    let v = spawn(&mut sim, &preset, 1.5);
    run(&mut sim, 60);
    sim.set_vehicle_input(v, throttle(0.7, 0.05));
    let mut out = Vec::new();
    for _ in 0..8 {
        run(&mut sim, 30);
        out.push(chassis(&sim, v).position);
    }
    out
}

#[test]
fn grid_walk_wheel_queries_match_the_full_scan() {
    for preset in [presets::sedan(), presets::monster_truck()] {
        let grid = course_run(false, preset.clone());
        let full = course_run(true, preset.clone());
        for (k, (a, b)) in grid.iter().zip(&full).enumerate() {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            eprintln!("{} t{k}: grid {a:?} full {b:?} Δ {d:e}", preset.name);
            assert!(
                d < 1e-4,
                "{}: grid walk diverges from full scan by {d}",
                preset.name
            );
        }
        assert!(
            grid.last().unwrap()[2] > 20.0,
            "{} drove the course",
            preset.name
        );
    }
}

#[test]
fn query_rate_extrapolates_contacts_between_probes() {
    let run_with = |interval: u32| {
        let mut sim = world();
        let mut desc = presets::sedan();
        desc.query_interval = interval;
        let v = spawn(&mut sim, &desc, 1.0);
        run(&mut sim, 120);
        let sag = chassis(&sim, v).position[1];
        drive(&mut sim, v, throttle(0.8, 0.0), 150);
        (
            sag,
            sim.vehicle_state(v).forward_speed,
            chassis(&sim, v).position,
        )
    };
    let (s1, v1, p1) = run_with(0);
    let (s4, v4, p4) = run_with(4);
    eprintln!("every step: sag {s1:.4} v {v1:.2} {p1:?}; every 4th: sag {s4:.4} v {v4:.2} {p4:?}");
    assert!((s1 - s4).abs() < 0.01, "same ride height");
    assert!((v1 - v4).abs() < 0.05 * v1.abs(), "same acceleration");
    assert!(p4[0].abs() < 0.2, "drives straight");
}

fn roll_pitch(c: &BodyState) -> (f32, glam::Vec3) {
    let q = glam::Quat::from_array(c.orientation);
    let up = q * glam::Vec3::Y;
    (up.y.clamp(-1.0, 1.0).acos().to_degrees(), up)
}

#[test]
fn tank_pivots_in_place_drives_and_turns_by_track_ratio() {
    let mut sim = world();
    let v = spawn(&mut sim, &presets::tank(), 2.5);
    run(&mut sim, 120);
    let st = sim.vehicle_state(v);
    // The raised sprocket and idler at each end of a track stay off the ground.
    assert_eq!(st.contacts, 10, "all road wheels down");
    let p0 = chassis(&sim, v).position;
    // Pivot: steer with no throttle counter-rotates the tracks.
    drive(
        &mut sim,
        v,
        VehicleInput {
            steer: 1.0,
            ..Default::default()
        },
        120,
    );
    let c = chassis(&sim, v);
    let st = sim.vehicle_state(v);
    let moved = ((c.position[0] - p0[0]).powi(2) + (c.position[2] - p0[2]).powi(2)).sqrt();
    eprintln!(
        "pivot: yaw {:.2} rad/s moved {moved:.2} m gear {} rpm {:.0} clutch {:.2} ω {:?} fL {:?}",
        st.yaw_rate,
        st.gear,
        st.engine_rpm,
        st.clutch,
        st.wheels.iter().map(|w| w.omega as i32).collect::<Vec<_>>(),
        st.wheels
            .iter()
            .map(|w| w.longitudinal_force as i32)
            .collect::<Vec<_>>()
    );
    assert!(
        st.yaw_rate < -0.2,
        "right pivot yaws right: {}",
        st.yaw_rate
    );
    assert!(moved < 1.0, "pivots in place: {moved}");
    // Straight line.
    drive(&mut sim, v, VehicleInput::default(), 60);
    drive(&mut sim, v, throttle(1.0, 0.0), 240);
    let st = sim.vehicle_state(v);
    eprintln!(
        "drive: speed {:.2} gear {} rpm {:.0}",
        st.forward_speed, st.gear, st.engine_rpm
    );
    assert!(st.forward_speed > 3.0, "tank drives: {}", st.forward_speed);
    // Steering while driving slows the inner track.
    drive(&mut sim, v, throttle(1.0, -1.0), 120);
    let st = sim.vehicle_state(v);
    eprintln!("turn: yaw {:.2} speed {:.2}", st.yaw_rate, st.forward_speed);
    assert!(st.yaw_rate > 0.05, "left steer turns left: {}", st.yaw_rate);
    assert!(roll_pitch(&chassis(&sim, v)).0 < 20.0);
    // Arcade skid steer: steer alone while rolling still rotates the hull, and in
    // reverse left steer still yaws left (the hull rotates, not the "car" path).
    drive(
        &mut sim,
        v,
        VehicleInput {
            steer: 1.0,
            ..Default::default()
        },
        90,
    );
    let st = sim.vehicle_state(v);
    eprintln!(
        "coast + steer: yaw {:.2} speed {:.2}",
        st.yaw_rate, st.forward_speed
    );
    assert!(
        st.yaw_rate < -0.2,
        "steer-only yaws right while moving: {}",
        st.yaw_rate
    );
    drive(
        &mut sim,
        v,
        VehicleInput {
            brake: 1.0,
            ..Default::default()
        },
        90,
    );
    drive(&mut sim, v, throttle(-1.0, 0.0), 180);
    let st = sim.vehicle_state(v);
    eprintln!("reverse: speed {:.2} gear {}", st.forward_speed, st.gear);
    assert!(
        st.forward_speed < -1.0,
        "tank reverses: {}",
        st.forward_speed
    );
    drive(&mut sim, v, throttle(-1.0, -1.0), 90);
    let st = sim.vehicle_state(v);
    eprintln!(
        "reverse + left: yaw {:.2} speed {:.2}",
        st.yaw_rate, st.forward_speed
    );
    assert!(
        st.yaw_rate > 0.2,
        "left steer yaws left in reverse too: {}",
        st.yaw_rate
    );
}

/// The tank as it was before track pods: wheel shells only, belly 1 m up.
fn tank_without_pods() -> VehicleDesc {
    let mut d = presets::tank();
    let t = d.tracks.as_mut().unwrap();
    t.left.pod = None;
    t.right.pod = None;
    let belly = d.parts.iter().position(|p| p.name == "belly").unwrap();
    d.parts[0].mass += d.parts.remove(belly).mass; // no belly plate
    d
}

/// Chassis-space pod boxes (centre, half extents) of the tank preset's tracks.
fn tank_pod_boxes() -> Vec<(glam::Vec3, glam::Vec3)> {
    let d = presets::tank().centered();
    let t = d.tracks.as_ref().unwrap();
    [&t.left, &t.right]
        .iter()
        .flat_map(|tr| {
            engine_vehicle::pack::track_pod_colliders(&d, tr, tr.pod.as_ref().unwrap(), 0)
        })
        .map(|c| {
            (
                glam::Vec3::from_array(c.local_pos),
                glam::Vec3::from_slice(&c.params[..3]),
            )
        })
        .collect()
}

struct DebrisRun {
    travelled: f32,
    speed: f32,
    max_tilt: f32,
    /// Deepest a crate got into a track run (m): caught between the road wheels.
    max_depth: f32,
}

/// Drive a tank at 5 m/s through rows of crates (half size `h`) in its track paths.
fn debris_run(desc: &VehicleDesc, h: f32) -> DebrisRun {
    let mut sim = world();
    let mut crates = Vec::new();
    for row in 0..8 {
        let z = 8.0 + 1.4 * row as f32;
        let jitter = if row % 2 == 0 { 0.12 } else { -0.12 };
        for x in [1.7f32, -1.7, 0.0] {
            let m = 1500.0 * 8.0 * h * h * h;
            crates.push(dyn_body(
                &mut sim,
                [x + jitter, h, z],
                m,
                [m * 4.0 * h * h / 6.0; 3],
                &[([0.0; 3], [h; 3], 0)],
            ));
        }
    }
    let v = spawn(&mut sim, desc, 2.5);
    run(&mut sim, 90);
    let p0 = chassis(&sim, v).position;
    let pods = tank_pod_boxes();
    let (mut max_depth, mut max_tilt) = (0.0f32, 0.0f32);
    for step in 0..600 {
        let e = 5.0 - sim.vehicle_state(v).forward_speed;
        sim.set_vehicle_input(v, throttle((e * 0.5).clamp(0.0, 1.0), 0.0));
        sim.step().expect("step");
        if step % 5 != 0 {
            continue;
        }
        let c = chassis(&sim, v);
        max_tilt = max_tilt.max(roll_pitch(&c).0);
        let q = glam::Quat::from_array(c.orientation);
        let o = glam::Vec3::from_array(c.position);
        for &b in &crates {
            let local = q.inverse() * (glam::Vec3::from_array(sim.read_body(b).position) - o);
            for (pc, half) in &pods {
                max_depth = max_depth.max(h - ((local - *pc).abs() - *half).max_element());
            }
        }
    }
    let c = chassis(&sim, v);
    let travelled = c.position[2] - p0[2];
    let speed = sim.vehicle_state(v).forward_speed;
    eprintln!("{} {} {:.2} m crates: travelled {travelled:.1} m, speed {speed:.2}, max tilt {max_tilt:.1}°, deepest in a track {max_depth:.2} m",
        desc.name, if desc.tracks.as_ref().unwrap().left.pod.is_some() { "pods" } else { "shells" }, 2.0 * h);
    DebrisRun {
        travelled,
        speed,
        max_tilt,
        max_depth,
    }
}

#[test]
fn tank_tracks_roll_over_debris_without_wedging_it_between_wheels() {
    // Crates in the track paths used to slip between the road-wheel shells and ride
    // inside the track; the pod is solid and its belt rolls the tank over them.
    for h in [0.11f32, 0.175] {
        let old = debris_run(&tank_without_pods(), h);
        let new = debris_run(&presets::tank(), h);
        assert!(
            new.max_depth < 0.08,
            "no crate inside a track run: {}",
            new.max_depth
        );
        assert!(
            new.travelled > 25.0,
            "drives through the crates: {}",
            new.travelled
        );
        assert!(new.speed > 3.0, "keeps moving: {}", new.speed);
        assert!(new.max_tilt < 25.0, "stays upright: {}", new.max_tilt);
        eprintln!("shells reached {:.2} m into a track", old.max_depth);
    }
}

/// Drive a tank at a step `height` tall; returns (how far past the step face the
/// chassis got, height gained).
fn step_climb(desc: &VehicleDesc, height: f32) -> (f32, f32) {
    let mut sim = world();
    let face = 9.0f32;
    static_box(
        &mut sim,
        [0.0, 0.5 * height, face + 200.0],
        [8.0, 0.5 * height, 200.0],
        [0.0, 0.0, 0.0, 1.0],
    );
    let v = spawn(&mut sim, desc, 2.5);
    run(&mut sim, 90);
    let y0 = chassis(&sim, v).position[1];
    drive(&mut sim, v, throttle(1.0, 0.0), 600);
    let c = chassis(&sim, v);
    let (past, rise) = (c.position[2] - face, c.position[1] - y0);
    eprintln!(
        "{} {} {height} m step: past the face {past:.2} m, rise {rise:.2} m, tilt {:.1}°",
        desc.name,
        if desc.tracks.as_ref().unwrap().left.pod.is_some() {
            "pods"
        } else {
            "shells"
        },
        roll_pitch(&c).0
    );
    (past, rise)
}

#[test]
fn tank_belt_climbs_a_step_taller_than_its_wheels() {
    // The road wheels reach 0.3 m: the old shell tank stalls at the face of
    // anything taller, while the pod's ramp and belt carry it up.
    for height in [0.5f32, 0.7] {
        let (old_past, _) = step_climb(&tank_without_pods(), height);
        let (past, rise) = step_climb(&presets::tank(), height);
        assert!(
            past > 5.0 && rise > 0.8 * height,
            "{height} m step: past the face {past}, rise {rise}"
        );
        assert!(
            old_past < 0.0,
            "shells alone stall at the face ({old_past})"
        );
    }
}

#[test]
fn idle_tank_belt_does_not_push_a_crate_it_touches() {
    // Belt speed is the track's rim speed: at rest it is plain friction.
    let mut sim = world();
    let v = spawn(&mut sim, &presets::tank(), 2.5);
    run(&mut sim, 90);
    let c = chassis(&sim, v);
    // A crate resting against the front of the left track run.
    let front = tank_pod_boxes()
        .iter()
        .map(|(pc, h)| pc.z + h.z)
        .fold(f32::MIN, f32::max);
    let crate_body = dyn_body(
        &mut sim,
        [1.7, 0.3, c.position[2] + front + 0.31],
        60.0,
        [60.0 * 0.18 / 3.0; 3],
        &[([0.0; 3], [0.3; 3], 0)],
    );
    run(&mut sim, 30);
    let c0 = chassis(&sim, v).position;
    let k0 = sim.read_body(crate_body).position;
    run(&mut sim, 180);
    let c1 = chassis(&sim, v).position;
    let k1 = sim.read_body(crate_body).position;
    let creep = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    eprintln!(
        "idle: tank creep {:.4} m, crate creep {:.4} m",
        creep(c0, c1),
        creep(k0, k1)
    );
    assert!(
        creep(c0, c1) < 0.02 && creep(k0, k1) < 0.02,
        "nothing creeps"
    );
}

#[test]
fn motorcycle_balances_and_leans_into_turns() {
    let mut sim = world();
    let v = spawn(&mut sim, &presets::motorcycle(), 1.2);
    run(&mut sim, 120);
    let (tilt, _) = roll_pitch(&chassis(&sim, v));
    eprintln!("parked tilt {tilt:.2}°");
    assert!(tilt < 5.0, "lean controller holds the bike up at rest");
    drive(&mut sim, v, throttle(0.6, 0.0), 300);
    let st = sim.vehicle_state(v);
    let c = chassis(&sim, v);
    let (tilt, _) = roll_pitch(&c);
    eprintln!(
        "straight: speed {:.2} tilt {tilt:.2}° pos {:?}",
        st.forward_speed, c.position
    );
    assert!(
        st.forward_speed > 8.0,
        "bike accelerates: {}",
        st.forward_speed
    );
    assert!(tilt < 8.0, "rides upright");
    // Left turn: the bike should lean left (+X is left) and keep going.
    drive(&mut sim, v, throttle(0.3, -0.5), 90);
    let st = sim.vehicle_state(v);
    let c = chassis(&sim, v);
    let (tilt, up) = roll_pitch(&c);
    let q = glam::Quat::from_array(c.orientation);
    let left = q * glam::Vec3::X;
    let lean_left = up.dot(glam::Vec3::Y.cross(q * glam::Vec3::Z).normalize());
    eprintln!(
        "turn: yaw {:.2} tilt {tilt:.1}° lean·left {lean_left:.3} left {left:?}",
        st.yaw_rate
    );
    assert!(st.yaw_rate > 0.1, "turns left: {}", st.yaw_rate);
    assert!(tilt > 3.0 && tilt < 50.0, "leans but does not fall: {tilt}");
    assert!(lean_left > 0.0, "leans into the turn (toward +X / left)");
    assert_eq!(st.contacts, 2, "both wheels on the road");
}

#[test]
fn crashing_into_light_crates_does_not_launch_the_vehicle() {
    // A stack of 40 kg crates. Wheels used to treat the crates as immovable ground and
    // the vehicle was thrown 300+ m up. Now tyre rows on crates are two-body and wheel
    // shells take side hits: the crates get shoved and the vehicle stays bounded.
    // Wheel loads stay physical: a wheel mount inside a crate used to read as a hit at
    // t = 0 along the suspension axis (full bump, 60× the truck's weight). The worst
    // real hit left is a tyre climbing a crate edge at ~4.5 m/s into the surface: the
    // 6 Hz bump damper alone gives ~176 kN, about 6× the monster truck's weight.
    for preset in [presets::monster_truck(), presets::sedan()] {
        let mut sim = world();
        sim.set_material(
            1,
            Material {
                friction: 0.6,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
        let mut crates = Vec::new();
        for y in 0..5 {
            for i in 0..(4 - (y >> 1)) {
                let body = sim.add_body(BodyState::dynamic(
                    glam::Vec3::new((i as f32 - 1.5) * 1.05, 0.5 + y as f32, 13.0),
                    glam::Quat::IDENTITY,
                    40.0,
                    glam::Vec3::splat(40.0 / 6.0),
                ));
                crates.push((
                    body,
                    glam::Vec3::new((i as f32 - 1.5) * 1.05, 0.5 + y as f32, 13.0),
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
        let v = spawn(&mut sim, &preset, 2.5);
        run(&mut sim, 60);
        let (mut max_v, mut max_w, mut max_y, mut max_fn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for _ in 0..240 {
            let e = 11.0 - sim.vehicle_state(v).forward_speed;
            sim.set_vehicle_input(v, throttle((e * 0.4).clamp(0.0, 1.0), 0.0));
            sim.step().expect("step");
            let c = chassis(&sim, v);
            max_v = max_v.max(glam::Vec3::from_array(c.linear_vel).length());
            max_w = max_w.max(glam::Vec3::from_array(c.angular_vel).length());
            max_y = max_y.max(c.position[1]);
            max_fn = sim
                .vehicle_state(v)
                .wheels
                .iter()
                .fold(max_fn, |m, w| m.max(w.suspension_force));
        }
        let weight = mass_of(&preset) * 10.0;
        let shoved = crates
            .iter()
            .map(|&(b, p0)| (glam::Vec3::from_array(sim.read_body(b).position) - p0).length())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{}: max |v| {max_v:.1} |ω| {max_w:.1} y {max_y:.2} peak wheel load {:.1}× weight, crates shoved up to {shoved:.1} m",
            preset.name,
            max_fn / weight
        );
        assert!(max_v < 16.0, "{}: thrown at {max_v} m/s", preset.name);
        assert!(max_w < 6.0, "{}: spun at {max_w} rad/s", preset.name);
        assert!(max_y < 3.0, "{}: launched to y {max_y}", preset.name);
        assert!(
            max_fn < 12.0 * weight,
            "{}: peak wheel load {:.1}× weight",
            preset.name,
            max_fn / weight
        );
        assert!(
            shoved > 0.5,
            "{}: the crates react ({shoved} m)",
            preset.name
        );
    }
}

#[test]
fn a_ledge_beside_the_wheel_mount_is_not_a_suspension_hit() {
    // A static ledge beside the front-left wheel at mount height: 0.5 m outboard of
    // the suspension axis, so the sphere cast (tyre radius 0.8) starts inside it, but
    // ~0.9 m from the wheel centre, so neither the tyre nor its shell touches it. Its
    // face is a side hit and the tyre stands on the ground below. Casts used to
    // report an origin-inside hit as t = 0 along the suspension axis, so the bump stop
    // fired straight up against a ledge the tyre never touched.
    let mut sim = world();
    let desc = presets::monster_truck();
    // Front-left mount (1.45, -0.1, 1.75) from the chassis, at y ≈ 1.6 once settled.
    static_box(
        &mut sim,
        [1.45 + 0.5 + 0.4, 1.95, 1.75],
        [0.4, 0.4, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    );
    let v = spawn(&mut sim, &desc, 2.0);
    let weight = mass_of(&desc) * 10.0;
    let (mut max_fn, mut max_y, mut drift) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..180 {
        sim.step().expect("step");
        let c = chassis(&sim, v);
        max_y = max_y.max(c.position[1]);
        drift = drift.max(c.position[0].abs().max(c.position[2].abs()));
        let st = sim.vehicle_state(v);
        max_fn = st
            .wheels
            .iter()
            .fold(max_fn, |m, w| m.max(w.suspension_force));
        let w = &st.wheels[0];
        assert!(
            w.ground_body != Some(1),
            "step {i}: the front-left tyre took the ledge as ground: {w:?}"
        );
    }
    let w = &sim.vehicle_state(v).wheels[0];
    assert!(
        w.in_contact && w.ground_body == Some(0) && w.contact_normal[1] > 0.99,
        "tyre on the ground: {w:?}"
    );
    eprintln!("ledge beside the mount: peak wheel load {:.2}× weight, max y {max_y:.2}, drift {drift:.3} m", max_fn / weight);
    assert!(
        max_fn < 1.5 * weight,
        "no bump-stop spike ({:.1}× weight)",
        max_fn / weight
    );
    assert!(
        max_y < 2.05 && drift < 0.05,
        "no kick off the ledge (max y {max_y}, drift {drift} m)"
    );
}

#[test]
fn handbrake_locks_the_rear_and_kicks_the_tail_out() {
    // Locked rear tyres slide along their slip velocity (mostly longitudinal), so they
    // lose lateral grip and the car rotates much faster than with steering alone.
    let heading_after = |handbrake: f32| {
        let mut sim = world();
        let v = spawn(&mut sim, &presets::sedan(), 1.0);
        run(&mut sim, 60);
        for _ in 0..400 {
            let e = 15.0 - sim.vehicle_state(v).forward_speed;
            sim.set_vehicle_input(v, throttle((e * 0.4).clamp(0.0, 1.0), 0.0));
            sim.step().expect("step");
            if e.abs() < 0.2 {
                break;
            }
        }
        sim.set_vehicle_input(
            v,
            VehicleInput {
                steer: -1.0,
                handbrake,
                ..Default::default()
            },
        );
        let mut heading = 0.0f32;
        for _ in 0..60 {
            sim.step().expect("step");
            heading += sim.vehicle_state(v).yaw_rate / 60.0;
        }
        let st = sim.vehicle_state(v);
        (
            heading.to_degrees(),
            st.wheels[2].omega.abs() + st.wheels[3].omega.abs(),
        )
    };
    let (plain, _) = heading_after(0.0);
    let (hand, rear_spin) = heading_after(1.0);
    eprintln!("1 s from 15 m/s, full lock: plain {plain:.1}°, handbrake {hand:.1}° (rear ω {rear_spin:.2})");
    assert!(rear_spin < 0.5, "handbrake locks the rear wheels");
    assert!(
        hand > 1.6 * plain,
        "handbrake turn rotates harder: {hand}° vs {plain}°"
    );
}

#[test]
fn motorcycle_holds_a_straight_line_under_full_throttle_after_a_turn() {
    // Play-test regression: after a slow turn, full throttle swapped the bike's ends
    // (wheelspin stripped the rear's side grip). Traction control and the
    // peak-normalised combined-slip blend keep it straight.
    let mut sim = world();
    let v = spawn(&mut sim, &presets::motorcycle(), 1.2);
    run(&mut sim, 90);
    drive(&mut sim, v, throttle(0.3, -0.6), 180);
    sim.set_vehicle_input(v, throttle(1.0, 0.0));
    let (mut max_yaw, mut max_lean) = (0.0f32, 0.0f32);
    for i in 0..480 {
        sim.step().expect("step");
        if i < 120 {
            continue; // straightening up
        }
        let st = sim.vehicle_state(v);
        max_yaw = max_yaw.max(st.yaw_rate.abs());
        max_lean = max_lean.max(roll_pitch(&chassis(&sim, v)).0);
    }
    let st = sim.vehicle_state(v);
    eprintln!(
        "straight: speed {:.1} gear {} max |yaw| {max_yaw:.2} max tilt {max_lean:.1}°",
        st.forward_speed, st.gear
    );
    assert!(st.forward_speed > 25.0, "accelerates: {}", st.forward_speed);
    assert!(max_yaw < 0.3, "holds its heading: |yaw| up to {max_yaw}");
    assert!(max_lean < 15.0, "stays upright: {max_lean}°");
}

/// A dynamic box body with one or more box colliders (material per collider).
fn dyn_body(
    sim: &mut WgpuSim,
    pos: [f32; 3],
    mass: f32,
    inertia: [f32; 3],
    boxes: &[([f32; 3], [f32; 3], u32)],
) -> u32 {
    let body = sim.add_body(BodyState::dynamic(
        glam::Vec3::from_array(pos),
        glam::Quat::IDENTITY,
        mass,
        glam::Vec3::from_array(inertia),
    ));
    for &(local, half, material) in boxes {
        sim.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material,
            generation: 0,
            local_pos: local,
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [
                half[0],
                half[1],
                half[2],
                (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
            ],
        });
    }
    body
}

#[test]
fn tyre_forces_push_back_on_a_dynamic_ground_body() {
    // A sedan accelerates on a heavy slab that slides on frictionless ice. The tyre
    // rows are two-body constraints, so the slab is pushed back inside the same
    // solve and the pair's total horizontal momentum stays ~zero.
    let mut sim = world();
    sim.set_material(
        1,
        Material {
            friction: 1.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    sim.set_material(
        2,
        Material {
            friction: 0.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    let slab_mass = 2500.0f32;
    let (hx, hz) = (6.0f32, 30.0f32);
    let slab = dyn_body(
        &mut sim,
        [0.0, 0.25, 0.0],
        slab_mass,
        [
            slab_mass * (0.25 + hz * hz) / 3.0,
            slab_mass * (hx * hx + hz * hz) / 3.0,
            slab_mass * (hx * hx + 0.25) / 3.0,
        ],
        // Grippy deck on top, frictionless skid underneath.
        &[
            ([0.0, 0.1, 0.0], [hx, 0.15, hz], 1),
            ([0.0, -0.1, 0.0], [hx, 0.15, hz], 2),
        ],
    );
    let desc = presets::sedan();
    let car_mass = mass_of(&desc);
    let v = sim
        .add_vehicle(&desc, [0.0, 2.0, -20.0], [0.0, 0.0, 0.0, 1.0])
        .expect("vehicle");
    run(&mut sim, 90);
    let car0 = chassis(&sim, v);
    let slab0 = sim.read_body(slab);
    eprintln!(
        "settled: car y {:.3} slab y {:.3} slab v {:?}",
        car0.position[1], slab0.position[1], slab0.linear_vel
    );
    drive(&mut sim, v, throttle(1.0, 0.0), 120);
    let car = chassis(&sim, v);
    let sl = sim.read_body(slab);
    let p_car = car_mass * car.linear_vel[2];
    let p_slab = slab_mass * sl.linear_vel[2];
    eprintln!(
        "after 2 s: car vz {:.2} slab vz {:.3} p_car {:.0} p_slab {:.0} sum {:.0} slab moved {:.2} m",
        car.linear_vel[2], sl.linear_vel[2], p_car, p_slab, p_car + p_slab, sl.position[2] - slab0.position[2]
    );
    assert!(car.linear_vel[2] > 3.0, "car accelerates on the slab");
    assert!(
        sl.linear_vel[2] < -0.5,
        "slab is pushed backwards: {}",
        sl.linear_vel[2]
    );
    assert!(
        (p_car + p_slab).abs() < 0.1 * p_car.abs(),
        "momentum conserved: {} vs {}",
        p_car + p_slab,
        p_car
    );
}

#[test]
fn a_parked_car_tips_a_seesaw_plank() {
    // The deck feels the car's weight in-step: a plank on a central pivot tips under
    // a car parked on one end (it used to see only a lagged, capped reaction).
    let mut sim = world();
    static_box(
        &mut sim,
        [0.0, 0.5, 0.0],
        [3.0, 0.5, 0.25],
        [0.0, 0.0, 0.0, 1.0],
    );
    let (hz, m) = (6.0f32, 300.0f32);
    let plank = dyn_body(
        &mut sim,
        [0.0, 1.15, 0.0],
        m,
        [m * hz * hz / 3.0, m * hz * hz / 3.0, m * 4.0 / 3.0],
        &[([0.0; 3], [2.0, 0.15, hz], 0)],
    );
    run(&mut sim, 30);
    let v = sim
        .add_vehicle(&presets::sedan(), [0.0, 2.6, 3.8], [0.0, 0.0, 0.0, 1.0])
        .expect("vehicle");
    let mut max_tilt = 0.0f32;
    for _ in 0..180 {
        sim.step().expect("step");
        let q = glam::Quat::from_array(sim.read_body(plank).orientation);
        let f = q * glam::Vec3::Z;
        max_tilt = max_tilt.max(-f.y.asin().to_degrees());
    }
    let c = chassis(&sim, v);
    eprintln!("plank tilt {max_tilt:.1}° car at {:?}", c.position);
    assert!(max_tilt > 10.0, "the car's end goes down: tilt {max_tilt}°");
}

#[test]
fn wheel_shell_shoves_a_crate_beside_the_tyre() {
    // The crate overlaps only the front-left tyre's outline: outside the body box and
    // clear of the wheel's ray line, so only the wheel shell can meet it. Without a
    // shell it would pass straight through the tyre.
    let mut sim = world();
    let crate_pos = [1.15f32, 0.2, 6.0];
    let crate_body = dyn_body(
        &mut sim,
        crate_pos,
        15.0,
        [15.0 * 0.32 / 3.0; 3],
        &[([0.0; 3], [0.3, 0.2, 0.3], 0)],
    );
    let v = spawn(&mut sim, &presets::sedan(), 1.0);
    run(&mut sim, 60);
    for _ in 0..150 {
        let e = 5.0 - sim.vehicle_state(v).forward_speed;
        sim.set_vehicle_input(v, throttle((e * 0.4).clamp(0.0, 1.0), 0.0));
        sim.step().expect("step");
    }
    let c = sim.read_body(crate_body);
    let car = chassis(&sim, v);
    eprintln!(
        "crate {:?} (from {crate_pos:?}), car z {:.2}",
        c.position, car.position[2]
    );
    assert!(
        car.position[2] > 8.0,
        "the car drove past the crate's start"
    );
    let moved =
        ((c.position[0] - crate_pos[0]).powi(2) + (c.position[2] - crate_pos[2]).powi(2)).sqrt();
    assert!(moved > 0.5, "the tyre shoved the crate ({moved} m)");
}

#[test]
fn a_car_parked_on_a_truck_loads_its_suspension() {
    // Vehicle on vehicle: the sedan's tyre rows act on the truck's chassis, so the
    // truck's own suspension carries both weights.
    let mut sim = world();
    // Converged loads, not the values frozen when the island falls asleep mid-bounce.
    let mut p = AvbdParams::default();
    p.iterations = 10;
    p.sleep_time = 1e6;
    sim.set_params(p);
    let truck = presets::monster_truck();
    let t = spawn(&mut sim, &truck, 2.0);
    run(&mut sim, 240);
    let alone: f32 = sim
        .vehicle_state(t)
        .wheels
        .iter()
        .map(|w| w.suspension_force)
        .sum();
    let roof = chassis(&sim, t).position[1];
    let sedan = presets::kart();
    let k = sim
        .add_vehicle(&sedan, [0.0, roof + 1.6, 0.0], [0.0, 0.0, 0.0, 1.0])
        .expect("vehicle");
    run(&mut sim, 360);
    let loaded: f32 = sim
        .vehicle_state(t)
        .wheels
        .iter()
        .map(|w| w.suspension_force)
        .sum();
    let ks = sim.vehicle_state(k);
    let kc = chassis(&sim, k);
    eprintln!(
        "truck wheel load alone {alone:.0} N, with the kart {loaded:.0} N (kart weight {:.0} N); kart contacts {} at y {:.2}, ground {:?}",
        mass_of(&sedan) * 10.0,
        ks.contacts,
        kc.position[1],
        ks.wheels.iter().map(|w| w.ground_body).collect::<Vec<_>>()
    );
    assert_eq!(ks.contacts, 4, "the kart stands on the truck");
    let gain = loaded - alone;
    let weight = mass_of(&sedan) * 10.0;
    assert!(
        (gain - weight).abs() < 0.1 * weight,
        "truck carries the kart: +{gain} N vs {weight} N"
    );
}

#[test]
fn air_control_pitches_an_airborne_kart() {
    let mut sim = world();
    let v = spawn(&mut sim, &presets::kart(), 20.0);
    sim.set_vehicle_input(v, throttle(1.0, 0.0));
    run(&mut sim, 30); // still falling (~1.6 m drop of 20 m)
    let c = chassis(&sim, v);
    let q = glam::Quat::from_array(c.orientation);
    let fwd = q * glam::Vec3::Z;
    eprintln!("airborne kart forward {fwd:?} ω {:?}", c.angular_vel);
    assert!(
        fwd.y < -0.05,
        "forward input pitches the nose down in the air: {fwd:?}"
    );
}

#[test]
fn raycast_batch_hits_ground_and_misses_sky() {
    let mut sim = world();
    let v = spawn(&mut sim, &presets::sedan(), 1.0);
    run(&mut sim, 10);
    let chassis_body = sim.vehicle_host().chassis_body(v);
    let ray = |o: [f32; 3], d: [f32; 3], ex: Option<u32>| engine_vehicle::Ray {
        origin: o,
        dir: d,
        max_dist: 10.0,
        exclude_body: ex,
    };
    let hits = sim.raycast_batch(&[
        ray([5.0, 3.0, 0.0], [0.0, -1.0, 0.0], None),
        ray([5.0, 3.0, 0.0], [0.0, 1.0, 0.0], None),
        ray([0.0, 5.0, 0.0], [0.0, -1.0, 0.0], None),
        ray([0.0, 5.0, 0.0], [0.0, -1.0, 0.0], Some(chassis_body)),
    ]);
    let ground = hits[0].expect("ground hit");
    assert!((ground.distance - 3.0).abs() < 1e-3 && (ground.normal[1] - 1.0).abs() < 1e-3);
    assert!(hits[1].is_none());
    let roof = hits[2].expect("roof hit");
    assert!(roof.distance < 4.7, "hits the car roof: {}", roof.distance);
    let through = hits[3].expect("ground under car");
    assert!(
        (through.distance - 5.0).abs() < 1e-3,
        "excluded chassis: {}",
        through.distance
    );
}

#[test]
fn every_preset_settles_drives_and_turns_without_blowing_up() {
    // A trailer stands on its own two wheels and a hitch (semi_pulls_its_trailer).
    for desc in presets::all()
        .into_iter()
        .filter(|d| !d.name.ends_with("_trailer"))
    {
        let mut sim = world();
        let v = spawn(&mut sim, &desc, 2.0);
        run(&mut sim, 150);
        let st = sim.vehicle_state(v);
        let c = chassis(&sim, v);
        // A tank's sprocket and idler (no suspension travel) ride above the ground.
        let n = desc
            .wheels
            .iter()
            .filter(|w| desc.tracks.is_none() || w.suspension_max > w.suspension_min)
            .count() as u32;
        assert_eq!(
            st.contacts, n,
            "{}: all wheels down after settling",
            desc.name
        );
        let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
        let weight = mass_of(&desc) * 10.0;
        assert!(
            (total - weight).abs() < 0.05 * weight,
            "{}: load {total} vs {weight}",
            desc.name
        );
        assert!(speed(&c) < 0.1, "{}: settled, v = {}", desc.name, speed(&c));
        drive(&mut sim, v, throttle(1.0, 0.0), 180);
        let fwd = sim.vehicle_state(v).forward_speed;
        drive(
            &mut sim,
            v,
            VehicleInput {
                brake: 0.6,
                ..Default::default()
            },
            60,
        );
        let braked = chassis(&sim, v);
        assert!(
            (glam::Quat::from_array(braked.orientation) * glam::Vec3::Y).y > 0.8,
            "{}: no stoppie",
            desc.name
        );
        drive(&mut sim, v, throttle(0.3, -0.4), 120);
        let st = sim.vehicle_state(v);
        let c = chassis(&sim, v);
        eprintln!(
            "{:>20}: settled y {:.3}, 3 s speed {:.1} m/s, turning yaw {:.2} gear {} rpm {:.0}",
            desc.name, c.position[1], fwd, st.yaw_rate, st.gear, st.engine_rpm
        );
        assert!(fwd > 5.0, "{}: accelerates ({fwd})", desc.name);
        assert!(
            st.yaw_rate > 0.05,
            "{}: left steer yaws left ({})",
            desc.name,
            st.yaw_rate
        );
        assert!(c.position.iter().all(|x| x.is_finite()));
        // Upright: chassis up stays near world up.
        let q = glam::Quat::from_array(c.orientation);
        assert!((q * glam::Vec3::Y).y > 0.8, "{}: stays upright", desc.name);
    }
}

// ---------------------------------------------------------------------------
// Breakable plank bridge (vehicle.html ?scene=bridge mirrors this layout).
// ---------------------------------------------------------------------------

/// Plank bridge: timber planks laid end to end across the gap between two
/// abutments. The end planks rest on the abutment seats (contact only: a simply
/// supported span); each seam is bolted with four ball joints at the plank
/// corners, top and bottom, so bending is carried as a tension/compression couple
/// through the bolts. Road top at `BRIDGE_DECK_Y`, travel along +Z from z < 0.
struct Bridge {
    bolts: Vec<u32>,
    /// Far end of the deck (z).
    end_z: f32,
}

const BRIDGE_DECK_Y: f32 = 3.0;
const BRIDGE_PLANKS: u32 = 5;
const PLANK_LEN: f32 = 2.6;
const PLANK_GAP: f32 = 0.005;
const PLANK_HALF: [f32; 3] = [2.2, 0.1, 0.5 * PLANK_LEN];
/// Light timber decking, 200 kg per metre of deck.
const PLANK_MASS: f32 = 200.0 * PLANK_LEN;
/// Initial bolt penalty (N/m): a stiff steel bolt, far above a plank's inertial
/// stiffness m/h² ≈ 2e6 N/m, so the seams hold from the first step instead of the
/// span sagging while β·|C| ramps the penalty up from the default 1e4.
const BOLT_PENALTY: f32 = 1.0e8;
/// Bolt rating. At rest the midspan bolts carry ~102 kN (M = w·x(L−x)/2 over a
/// 0.32 m couple: 107 kN analytic). Crossing at 2.5 m/s a 1.3 t sedan peaks near
/// 205 kN, a 3 t monster truck near 320 kN and a 4.6 t tank near 450 kN
/// (`bridge_probe_unbreakable_bolt_forces`).
const BOLT_BREAK_FORCE: f32 = 255.0e3;
/// Colored Gauss–Seidel needs ~3 s of 10-iteration steps to converge the deck's
/// bending, longer than a fast crossing: at 10 a truck above ~9 m/s crossed before
/// the bolts felt its weight. 20 iterations track it (cheap for six bodies).
const BRIDGE_ITERATIONS: u32 = 20;

fn bridge(sim: &mut WgpuSim, break_force: f32) -> Bridge {
    let (hx, hy, hz) = (PLANK_HALF[0], PLANK_HALF[1], PLANK_HALF[2]);
    let pitch = PLANK_LEN + PLANK_GAP;
    let end_z = BRIDGE_PLANKS as f32 * pitch - PLANK_GAP;
    let (seat, road_half_len) = (0.5f32, 30.0f32);
    let road_x = hx + 0.3;
    let y = BRIDGE_DECK_Y;
    let id = [0.0, 0.0, 0.0, 1.0];
    // Approach roads (embankments) at deck height, and the lower seats the end
    // planks rest on.
    static_box(
        sim,
        [0.0, 0.5 * y, -PLANK_GAP - road_half_len],
        [road_x, 0.5 * y, road_half_len],
        id,
    );
    static_box(
        sim,
        [0.0, 0.5 * y, end_z + PLANK_GAP + road_half_len],
        [road_x, 0.5 * y, road_half_len],
        id,
    );
    let seat_top = y - 2.0 * hy;
    static_box(
        sim,
        [0.0, 0.5 * seat_top, 0.5 * (seat - PLANK_GAP)],
        [road_x, 0.5 * seat_top, 0.5 * (seat + PLANK_GAP)],
        id,
    );
    static_box(
        sim,
        [0.0, 0.5 * seat_top, end_z - 0.5 * (seat - PLANK_GAP)],
        [road_x, 0.5 * seat_top, 0.5 * (seat + PLANK_GAP)],
        id,
    );
    let inertia = [
        PLANK_MASS * (hy * hy + hz * hz) / 3.0,
        PLANK_MASS * (hx * hx + hz * hz) / 3.0,
        PLANK_MASS * (hx * hx + hy * hy) / 3.0,
    ];
    let planks: Vec<u32> = (0..BRIDGE_PLANKS)
        .map(|i| {
            dyn_body(
                sim,
                [0.0, y - hy, i as f32 * pitch + hz],
                PLANK_MASS,
                inertia,
                &[([0.0; 3], PLANK_HALF, 0)],
            )
        })
        .collect();
    let mut bolts = Vec::new();
    for w in planks.windows(2) {
        for (bx, by) in [
            (-0.8 * hx, -0.8 * hy),
            (0.8 * hx, -0.8 * hy),
            (-0.8 * hx, 0.8 * hy),
            (0.8 * hx, 0.8 * hy),
        ] {
            // Anchors meet in the middle of the seam gap.
            // A D6 ball bolt whose penalty starts at BOLT_PENALTY (the planks'
            // m/h² is far above the default kStart, which would let the span sag).
            let mut bolt = engine_schema::JointDesc::ball(
                w[0],
                w[1],
                glam::Vec3::new(bx, by, hz + 0.5 * PLANK_GAP),
                glam::Vec3::new(bx, by, -hz - 0.5 * PLANK_GAP),
            )
            .with_break(
                if break_force > 0.0 {
                    break_force
                } else {
                    engine_schema::JOINT_BIG
                },
                engine_schema::JOINT_BIG,
            )
            .to_gpu();
            bolt.penalty[..3].fill(BOLT_PENALTY);
            bolts.push(sim.add_joint(bolt));
        }
    }
    Bridge { bolts, end_z }
}

struct Crossing {
    /// Largest bolt force seen (N), sampled every 5 steps.
    max_bolt: f32,
    breaks: Vec<engine_schema::JointBreak>,
    /// Lowest chassis height over the run.
    min_y: f32,
    /// Chassis height on the approach road (ride height above the deck level).
    ride_y: f32,
    chassis: [f32; 3],
    end_z: f32,
}

/// Let the deck settle, then drive `desc` across holding `speed` (m/s) and stop on
/// the far road (or stop early once the vehicle has fallen through). Fast runs
/// start at the far end of the 30 m approach road to get up to speed.
fn cross_bridge(desc: &VehicleDesc, break_force: f32, speed: f32) -> Crossing {
    let mut sim = world();
    let mut p = AvbdParams::default();
    p.iterations = BRIDGE_ITERATIONS;
    sim.set_params(p);
    let br = bridge(&mut sim, break_force);
    run(&mut sim, 300);
    let start_z = if speed > 4.0 { -27.0 } else { -12.0 };
    let v = sim
        .add_vehicle(
            desc,
            [0.0, BRIDGE_DECK_Y + 1.5, start_z],
            [0.0, 0.0, 0.0, 1.0],
        )
        .expect("vehicle");
    run(&mut sim, 60);
    let (mut max_bolt, mut min_y) = (0.0f32, f32::MAX);
    let ride_y = chassis(&sim, v).position[1];
    let mut breaks = sim.poll_joint_breaks();
    for step in 0..1200 {
        let c = chassis(&sim, v);
        min_y = min_y.min(c.position[1]);
        let input = if c.position[2] > br.end_z + 6.0 {
            VehicleInput {
                brake: 1.0,
                ..Default::default()
            }
        } else {
            let e = speed - sim.vehicle_state(v).forward_speed;
            VehicleInput {
                forward: (e * 0.4).clamp(0.0, 1.0),
                brake: (-e * 0.5).clamp(0.0, 1.0),
                ..Default::default()
            }
        };
        sim.set_vehicle_input(v, input);
        sim.step().expect("step");
        breaks.extend(sim.poll_joint_breaks());
        if step % 5 == 0 {
            let joints = sim.download_joints();
            let f = br
                .bolts
                .iter()
                .map(|&j| glam::Vec3::from_slice(&joints[j as usize].lambda[..3]).length())
                .fold(0.0, f32::max);
            max_bolt = max_bolt.max(f);
            if step % 30 == 0 && std::env::var("BRIDGE_TRACE").is_ok() {
                eprintln!(
                    "  t {:.2} z {:.2} y {:.2} v {:.2} broken {} max bolt {:.0} kN",
                    step as f32 / 60.0,
                    c.position[2],
                    c.position[1],
                    sim.vehicle_state(v).forward_speed,
                    breaks.len(),
                    f / 1000.0
                );
            }
        }
        if c.position[1] < ride_y - 2.0 {
            run(&mut sim, 90); // let it land
            break;
        }
    }
    let c = chassis(&sim, v);
    eprintln!(
        "{}: max bolt {:.0} kN, {} bolts broke, min chassis y {:.2}, final {:?}",
        desc.name,
        max_bolt / 1000.0,
        breaks.len(),
        min_y,
        c.position
    );
    Crossing {
        max_bolt,
        breaks,
        min_y: min_y.min(c.position[1]),
        ride_y,
        chassis: c.position,
        end_z: br.end_z,
    }
}

#[test]
fn a_sedan_crosses_the_plank_bridge_and_the_bolts_hold() {
    let desc = presets::sedan();
    let r = cross_bridge(&desc, BOLT_BREAK_FORCE, 2.5);
    assert!(
        r.breaks.is_empty(),
        "no bolt breaks under a sedan: {:?}",
        r.breaks
    );
    assert!(
        r.max_bolt > 150.0e3,
        "the deck feels the car (two-way tyre rows): {} N",
        r.max_bolt
    );
    assert!(
        r.max_bolt < 0.9 * BOLT_BREAK_FORCE,
        "margin below the rating: {} N",
        r.max_bolt
    );
    assert!(
        r.chassis[2] > r.end_z + 3.0,
        "reached the far road: z {}",
        r.chassis[2]
    );
    assert!(
        r.min_y > r.ride_y - 0.1,
        "rode level across the deck: min y {} vs {}",
        r.min_y,
        r.ride_y
    );
}

#[test]
fn a_monster_truck_snaps_the_bolts_and_falls_through() {
    let desc = presets::monster_truck();
    let r = cross_bridge(&desc, BOLT_BREAK_FORCE, 2.5);
    assert!(!r.breaks.is_empty(), "the truck overloads the bolts");
    for b in &r.breaks {
        assert!(
            b.force > BOLT_BREAK_FORCE,
            "each break exceeded the rating: {b:?}"
        );
    }
    assert!(
        r.chassis[2] < r.end_z,
        "it never reached the far road: z {}",
        r.chassis[2]
    );
    // The deck is 3 m up; the truck lands on the fallen planks.
    assert!(
        r.chassis[1] < r.ride_y - 2.0,
        "the truck fell through the deck: y {} (ride {})",
        r.chassis[1],
        r.ride_y
    );
}

#[test]
#[ignore]
fn bridge_probe_unbreakable_bolt_forces() {
    for desc in [presets::sedan(), presets::monster_truck(), presets::tank()] {
        cross_bridge(&desc, 0.0, 2.5);
    }
}

#[test]
fn a_fast_monster_truck_breaks_the_bridge_but_makes_it_over() {
    // At 12 m/s the truck is past midspan when the bolts let go: the deck folds
    // behind it and its momentum carries it onto the far road.
    let r = cross_bridge(&presets::monster_truck(), BOLT_BREAK_FORCE, 12.0);
    assert!(
        r.breaks.len() >= 8,
        "the bridge still breaks: {} bolts",
        r.breaks.len()
    );
    assert!(
        r.chassis[2] > r.end_z + 3.0,
        "made it to the far road: z {}",
        r.chassis[2]
    );
    assert!(
        (r.chassis[1] - r.ride_y).abs() < 0.2,
        "on its wheels at road level: y {} vs {}",
        r.chassis[1],
        r.ride_y
    );
}

#[test]
#[ignore]
fn bridge_probe_truck_speed() {
    let speeds: Vec<f32> = std::env::var("SPEEDS")
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![5.0, 8.0, 10.0, 12.0, 14.0]);
    for v in speeds {
        let r = cross_bridge(&presets::monster_truck(), BOLT_BREAK_FORCE, v);
        eprintln!(
            "speed {v}: breaks {}, final {:?} (end_z {:.2}), ride {:.2}",
            r.breaks.len(),
            r.chassis,
            r.end_z,
            r.ride_y
        );
    }
}

#[test]
fn semi_pulls_its_trailer() {
    let mut sim = world();
    let tractor = presets::vibe_semi();
    let trailer = presets::vibe_semi_trailer();
    // Both in design space, lifted 0.6 m: each body sits at its own centre of
    // mass, and the hitch is the same design point in each.
    let (ov, ot) = (tractor.center_of_mass(), trailer.center_of_mass());
    let v = sim
        .add_vehicle(&tractor, [ov[0], ov[1] + 0.6, ov[2]], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    let t = sim
        .add_vehicle(&trailer, [ot[0], ot[1] + 0.6, ot[2]], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    let (cv, ct) = (
        sim.vehicle_host().chassis_body(v),
        sim.vehicle_host().chassis_body(t),
    );
    let h = glam::Vec3::from_array(presets::VIBE_SEMI_HITCH);
    sim.add_joint_desc(&engine_schema::JointDesc::ball(
        cv,
        ct,
        h - glam::Vec3::from_array(ov),
        h - glam::Vec3::from_array(ot),
    ));
    run(&mut sim, 240);
    let st = sim.vehicle_state(t);
    assert_eq!(st.contacts, 2, "trailer on its wheels");
    let z0 = sim.read_body(ct).position[2];
    drive(&mut sim, v, throttle(1.0, 0.0), 360);
    let speed = sim.vehicle_state(v).forward_speed;
    let tb = sim.read_body(ct);
    let gap = (glam::Vec3::from_array(sim.read_body(cv).position)
        - glam::Vec3::from_array(tb.position))
    .length();
    eprintln!(
        "semi {speed:.1} m/s, trailer moved {:.1} m, chassis gap {gap:.2} m",
        tb.position[2] - z0
    );
    assert!(speed > 5.0, "the semi pulls ({speed} m/s)");
    assert!(tb.position[2] - z0 > 10.0, "the trailer follows");
    assert!(
        (glam::Quat::from_array(tb.orientation) * glam::Vec3::Y).y > 0.9,
        "upright"
    );
}
