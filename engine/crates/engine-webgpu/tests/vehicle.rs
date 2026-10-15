//! Vehicle controller envelopes on the native wgpu host (same WGSL as Chrome).
//!
//! Analytic checks: static load balance, settling, straight-line drive, braking,
//! steering direction, parking without creep, severing parts and wheels.

use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_vehicle::{presets, FunctionalState, VehicleDesc, VehicleHandle, VehicleInput, WheelDesc};
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
        params: [half[0], half[1], half[2], (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt()],
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
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    static_box(&mut sim, [0.0, -5.0, 0.0], [400.0, 5.0, 400.0], [0.0, 0.0, 0.0, 1.0]);
    sim
}

fn spawn(sim: &mut WgpuSim, desc: &VehicleDesc, y: f32) -> VehicleHandle {
    sim.add_vehicle(desc, [0.0, y, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle")
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
    eprintln!("chassis y {:.4} v {:?} w {:?}", c.position[1], c.linear_vel, c.angular_vel);
    for (i, w) in st.wheels.iter().enumerate() {
        eprintln!(
            "wheel {i}: contact {} L {:.3} Fs {:.1} pos {:?} omega {:.3}",
            w.in_contact, w.suspension_length, w.suspension_force, w.position, w.omega
        );
    }
    assert_eq!(st.contacts, 4, "all four wheels on the ground");
    let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    let weight = mass_of(&desc) * 10.0;
    assert!((total - weight).abs() < 0.03 * weight, "suspension carries the weight: {total} vs {weight}");
    assert!(speed(&c) < 0.05, "chassis at rest: {:?}", c.linear_vel);
    for w in &st.wheels {
        assert!(w.suspension_length > 0.15 && w.suspension_length < 0.45, "travel within limits: {}", w.suspension_length);
        // Wheel bottoms touch the ground.
        assert!((w.position[1] - 0.34).abs() < 0.02, "wheel centre at radius height: {}", w.position[1]);
    }
}

fn drive(sim: &mut WgpuSim, v: VehicleHandle, input: VehicleInput, steps: u32) {
    sim.set_vehicle_input(v, input);
    run(sim, steps);
}

fn throttle(forward: f32, steer: f32) -> VehicleInput {
    VehicleInput { forward, steer, ..Default::default() }
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
            st.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>(),
            st.wheels.iter().map(|w| w.longitudinal_force as i32).collect::<Vec<_>>(),
            st.wheels.iter().map(|w| w.omega).collect::<Vec<_>>()
        );
    }
    let st = sim.vehicle_state(v);
    let c = chassis(&sim, v);
    assert!(st.forward_speed > 15.0, "reaches highway speed in 6 s: {}", st.forward_speed);
    assert!(max_gear >= 3, "automatic box shifts up: {max_gear}");
    assert!(st.engine_rpm < 6801.0);
    assert!(c.position[0].abs() < 0.5, "drives straight: x drift {}", c.position[0]);
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
    eprintln!("turning: yaw {:.3} pos {:?} speed {:.2}", st.yaw_rate, c.position, st.forward_speed);
    assert!(st.yaw_rate < -0.1, "right steer yaws right (negative about +Y): {}", st.yaw_rate);
    assert!(c.position[0] < -0.5, "heads to the right (-X): {}", c.position[0]);
    // Brake to a stop.
    drive(&mut sim, v, VehicleInput { brake: 1.0, ..Default::default() }, 240);
    let st = sim.vehicle_state(v);
    eprintln!("braked: speed {:.3}", st.forward_speed);
    assert!(st.forward_speed.abs() < 0.05, "stopped: {}", st.forward_speed);
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
    let v = sim.add_vehicle(&presets::sedan(), [0.0, 1.2, 0.0], rot).expect("vehicle");
    sim.set_vehicle_input(v, VehicleInput { handbrake: 1.0, brake: 1.0, ..Default::default() });
    run(&mut sim, 120);
    let start = chassis(&sim, v).position;
    run(&mut sim, 300);
    let end = chassis(&sim, v);
    let drift = ((end.position[0] - start[0]).powi(2) + (end.position[2] - start[2]).powi(2)).sqrt();
    eprintln!("slope drift {:.5} m over 5 s; sleeping {}", drift, end.is_sleeping());
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
    eprintln!("omega {omega:.2} wheel body w {:?} v {:?} flags {}", wheel.angular_vel, wheel.linear_vel, wheel.flags);
    assert!(wheel.is_dynamic(), "wheel became a rigid body");
    let spin = wheel.angular_vel[0];
    assert!((spin - omega).abs() < 0.15 * omega.abs() + 0.5, "spin carried over: {spin} vs {omega}");
    assert!(speed(&wheel) > 0.5 * speed(&c0), "wheel moves with the car");
    assert_eq!(st.wheel_mask & 1, 0, "wheel 0 removed from the vehicle");
    let m0 = 1.0 / c0.inv_mass;
    let m1 = 1.0 / c1.inv_mass;
    assert!((m0 - m1 - desc.wheels[0].mass).abs() < 0.05, "chassis mass drops by the wheel: {m0} -> {m1}");
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
    assert!(!st.engine_connected, "engine part gone → engine disconnected");
    assert!(st.parts_lost & (1 << engine) != 0);
    let part_body = sim.read_body(sim.vehicle_host().part_body(v, engine));
    assert!(part_body.is_dynamic(), "engine block is a free rigid body");
    drive(&mut sim, v, throttle(1.0, 0.0), 120);
    let st = sim.vehicle_state(v);
    eprintln!("no-engine speed {:.3}", st.forward_speed);
    assert!(st.forward_speed.abs() < 0.3, "no drive without an engine: {}", st.forward_speed);
}

#[test]
fn losing_the_driveshaft_cuts_drive_even_with_overrides() {
    let mut sim = world();
    let desc = presets::destructible_sedan();
    let v = spawn(&mut sim, &desc, 1.0);
    run(&mut sim, 60);
    let shaft = desc.parts.iter().position(|p| p.name == "driveshaft").unwrap() as u32;
    sim.disable_part(v, shaft);
    drive(&mut sim, v, throttle(1.0, 0.0), 120);
    let st = sim.vehicle_state(v);
    assert_eq!(st.drive_mask, 0, "rear wheels lost drive");
    assert!(st.forward_speed.abs() < 0.3, "no drive: {}", st.forward_speed);
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
    let extra = WheelDesc { position: [0.82, -0.05, 0.0], ..desc.wheels[2].clone() };
    let idx = sim.add_wheel(v, &extra).expect("spare slot");
    assert_eq!(idx, 4);
    run(&mut sim, 120);
    let st = sim.vehicle_state(v);
    let m1 = 1.0 / chassis(&sim, v).inv_mass;
    assert_eq!(st.wheels.len(), 5);
    assert!(st.wheels[4].in_contact, "new wheel reaches the ground");
    assert!((m1 - m0 - extra.mass).abs() < 0.05, "mass added: {m0} -> {m1}");
    let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    let weight = m1 * 10.0;
    assert!((total - weight).abs() < 0.05 * weight, "five wheels carry the weight: {total} vs {weight}");
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
    let snap = sim.snapshot();
    let chassis_body = sim.vehicle_host().chassis_body(v);
    let mut first = Vec::new();
    let mut events = Vec::new();
    for k in 0..3 {
        drive(&mut sim, v, throttle(1.0, if k == 1 { 0.5 } else { 0.0 }), 40);
        if k == 1 {
            sim.detach_wheel(v, 2);
            sim.disable_part(v, 3);
        }
        events.extend(sim.poll_vehicle_events());
        first.push(sim.read_body(chassis_body));
    }
    assert!(events.iter().any(|e| matches!(e, engine_vehicle::VehicleEvent::WheelDetached { wheel: 2, .. })), "{events:?}");
    assert!(events.iter().any(|e| matches!(e, engine_vehicle::VehicleEvent::PartLost { part: 3, .. })), "{events:?}");
    assert!(events.iter().any(|e| matches!(e, engine_vehicle::VehicleEvent::FunctionalStateChanged { engine_connected: false, .. })));

    sim.restore(&snap);
    let mut second = Vec::new();
    for k in 0..3 {
        drive(&mut sim, v, throttle(1.0, if k == 1 { 0.5 } else { 0.0 }), 40);
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
    static_box(&mut sim, [0.0, 0.075, 10.0], [6.0, 0.075, 1.5], [0.0, 0.0, 0.0, 1.0]);
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
            eprintln!("on kerb: z {:.2} y {:.3} contacts {} L {:?}", c.position[2], c.position[1], st.contacts,
                st.wheels.iter().map(|w| (w.suspension_length * 100.0).round() / 100.0).collect::<Vec<_>>());
        }
    }
    let c = chassis(&sim, v);
    eprintln!("kerb: settled {settled:.3} peak {peak:.3} end {:?}", c.position);
    assert!(c.position[2] > 14.0, "drove over the kerb: z {}", c.position[2]);
    assert!(peak > settled + 0.08 && peak < settled + 0.5, "rode up onto the kerb: {settled} → {peak}");
    let q = glam::Quat::from_array(c.orientation);
    assert!((q * glam::Vec3::Y).y > 0.95, "level after the kerb");
}

fn obstacle_course(sim: &mut WgpuSim) {
    static_box(sim, [0.0, 0.075, 12.0], [6.0, 0.075, 1.5], [0.0, 0.0, 0.0, 1.0]);
    let a = -0.15f32;
    static_box(sim, [3.0, 0.4, 25.0], [3.0, 0.3, 5.0], [(a * 0.5).sin(), 0.0, 0.0, (a * 0.5).cos()]);
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
            assert!(d < 1e-4, "{}: grid walk diverges from full scan by {d}", preset.name);
        }
        assert!(grid.last().unwrap()[2] > 20.0, "{} drove the course", preset.name);
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
        (sag, sim.vehicle_state(v).forward_speed, chassis(&sim, v).position)
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
    drive(&mut sim, v, VehicleInput { steer: 1.0, ..Default::default() }, 120);
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
        st.wheels.iter().map(|w| w.longitudinal_force as i32).collect::<Vec<_>>()
    );
    assert!(st.yaw_rate < -0.2, "right pivot yaws right: {}", st.yaw_rate);
    assert!(moved < 1.0, "pivots in place: {moved}");
    // Straight line.
    drive(&mut sim, v, VehicleInput::default(), 60);
    drive(&mut sim, v, throttle(1.0, 0.0), 240);
    let st = sim.vehicle_state(v);
    eprintln!("drive: speed {:.2} gear {} rpm {:.0}", st.forward_speed, st.gear, st.engine_rpm);
    assert!(st.forward_speed > 3.0, "tank drives: {}", st.forward_speed);
    // Steering while driving slows the inner track.
    drive(&mut sim, v, throttle(1.0, -1.0), 120);
    let st = sim.vehicle_state(v);
    eprintln!("turn: yaw {:.2} speed {:.2}", st.yaw_rate, st.forward_speed);
    assert!(st.yaw_rate > 0.05, "left steer turns left: {}", st.yaw_rate);
    assert!(roll_pitch(&chassis(&sim, v)).0 < 20.0);
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
    eprintln!("straight: speed {:.2} tilt {tilt:.2}° pos {:?}", st.forward_speed, c.position);
    assert!(st.forward_speed > 8.0, "bike accelerates: {}", st.forward_speed);
    assert!(tilt < 8.0, "rides upright");
    // Left turn: the bike should lean left (+X is left) and keep going.
    drive(&mut sim, v, throttle(0.3, -0.5), 90);
    let st = sim.vehicle_state(v);
    let c = chassis(&sim, v);
    let (tilt, up) = roll_pitch(&c);
    let q = glam::Quat::from_array(c.orientation);
    let left = q * glam::Vec3::X;
    let lean_left = up.dot(glam::Vec3::Y.cross(q * glam::Vec3::Z).normalize()) ;
    eprintln!("turn: yaw {:.2} tilt {tilt:.1}° lean·left {lean_left:.3} left {left:?}", st.yaw_rate);
    assert!(st.yaw_rate > 0.1, "turns left: {}", st.yaw_rate);
    assert!(tilt > 3.0 && tilt < 50.0, "leans but does not fall: {tilt}");
    assert!(lean_left > 0.0, "leans into the turn (toward +X / left)");
    assert_eq!(st.contacts, 2, "both wheels on the road");
}

#[test]
fn crashing_into_light_crates_does_not_launch_the_vehicle() {
    // A stack of 40 kg crates: sphere-cast wheels that climb into it used to read full
    // compression and answer with MN bump-stop loads (vehicle thrown 300+ m up).
    for preset in [presets::monster_truck(), presets::sedan()] {
        let mut sim = world();
        sim.set_material(1, Material { friction: 0.6, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
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
            max_fn = sim.vehicle_state(v).wheels.iter().fold(max_fn, |m, w| m.max(w.suspension_force));
        }
        let weight = mass_of(&preset) * 10.0;
        eprintln!("{}: max |v| {max_v:.1} |ω| {max_w:.1} y {max_y:.2} Fn {:.1}× weight", preset.name, max_fn / weight);
        assert!(max_v < 16.0, "{}: thrown at {max_v} m/s", preset.name);
        assert!(max_w < 6.0, "{}: spun at {max_w} rad/s", preset.name);
        assert!(max_y < 6.0, "{}: launched to y {max_y}", preset.name);
        assert!(max_fn < 4.0 * weight, "{}: wheel load {max_fn} N", preset.name);
    }
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
    assert!(fwd.y < -0.05, "forward input pitches the nose down in the air: {fwd:?}");
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
    assert!((through.distance - 5.0).abs() < 1e-3, "excluded chassis: {}", through.distance);
}

#[test]
fn every_preset_settles_drives_and_turns_without_blowing_up() {
    for desc in presets::all() {
        let mut sim = world();
        let v = spawn(&mut sim, &desc, 2.0);
        run(&mut sim, 150);
        let st = sim.vehicle_state(v);
        let c = chassis(&sim, v);
        // A tank's sprocket and idler (no suspension travel) ride above the ground.
        let n = desc.wheels.iter().filter(|w| desc.tracks.is_none() || w.suspension_max > w.suspension_min).count() as u32;
        assert_eq!(st.contacts, n, "{}: all wheels down after settling", desc.name);
        let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
        let weight = mass_of(&desc) * 10.0;
        assert!((total - weight).abs() < 0.05 * weight, "{}: load {total} vs {weight}", desc.name);
        assert!(speed(&c) < 0.1, "{}: settled, v = {}", desc.name, speed(&c));
        drive(&mut sim, v, throttle(1.0, 0.0), 180);
        let fwd = sim.vehicle_state(v).forward_speed;
        drive(&mut sim, v, VehicleInput { brake: 0.6, ..Default::default() }, 60);
        let braked = chassis(&sim, v);
        assert!((glam::Quat::from_array(braked.orientation) * glam::Vec3::Y).y > 0.8, "{}: no stoppie", desc.name);
        drive(&mut sim, v, throttle(0.3, -0.4), 120);
        let st = sim.vehicle_state(v);
        let c = chassis(&sim, v);
        eprintln!(
            "{:>20}: settled y {:.3}, 3 s speed {:.1} m/s, turning yaw {:.2} gear {} rpm {:.0}",
            desc.name, c.position[1], fwd, st.yaw_rate, st.gear, st.engine_rpm
        );
        assert!(fwd > 5.0, "{}: accelerates ({fwd})", desc.name);
        assert!(st.yaw_rate > 0.05, "{}: left steer yaws left ({})", desc.name, st.yaw_rate);
        assert!(c.position.iter().all(|x| x.is_finite()));
        // Upright: chassis up stays near world up.
        let q = glam::Quat::from_array(c.orientation);
        assert!((q * glam::Vec3::Y).y > 0.8, "{}: stays upright", desc.name);
    }
}
