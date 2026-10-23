//! Vehicle mount breaking (`MountDesc`, `vehicle_loads`) on the native wgpu host.
//!
//! A part's mount carries what the part's own motion needs minus what its
//! contacts supply. At rest that is its weight; a part that is struck carries the
//! blow, and past its limit it leaves the chassis before the next step.

use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_vehicle::{presets, MountDesc, MountLoad, VehicleDesc, VehicleHandle, VehicleInput};
use engine_webgpu::WgpuSim;

const G: f32 = 10.0; // AvbdParams::default() gravity

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

fn add_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3], rot: [f32; 4], inv_mass: f32, inv_i: [f32; 3]) -> u32 {
    let body = sim.add_body(BodyState {
        position: pos,
        inv_mass,
        orientation: rot,
        inv_inertia: inv_i,
        flags: if inv_mass > 0.0 { 1 | 8 } else { 0 },
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

/// A free sphere of `mass` at `pos` moving at `vel`.
fn ball(sim: &mut WgpuSim, pos: [f32; 3], radius: f32, mass: f32, vel: [f32; 3]) -> u32 {
    let i = 0.4 * mass * radius * radius;
    let mut b = BodyState::dynamic(glam::Vec3::from_array(pos), glam::Quat::IDENTITY, mass, glam::Vec3::splat(i));
    b.linear_vel = vel;
    let body = sim.add_body(b);
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Sphere as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [radius, 0.0, 0.0, radius],
    });
    body
}

fn world() -> WgpuSim {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    add_box(&mut sim, [0.0, -5.0, 0.0], [400.0, 5.0, 400.0], [0.0, 0.0, 0.0, 1.0], 0.0, [0.0; 3]);
    sim
}

fn run(sim: &mut WgpuSim, steps: u32) {
    for _ in 0..steps {
        sim.step().expect("step");
    }
}

fn part_loads(sim: &WgpuSim, v: VehicleHandle) -> Vec<MountLoad> {
    sim.vehicle_mount_loads(v).into_iter().filter(|l| !l.wheel).collect()
}

fn wheel_loads(sim: &WgpuSim, v: VehicleHandle) -> Vec<MountLoad> {
    sim.vehicle_mount_loads(v).into_iter().filter(|l| l.wheel).collect()
}

/// The destructible sedan with every mount finite but far out of reach, so the
/// loads are measured and nothing breaks.
fn measured_sedan() -> VehicleDesc {
    let mut d = presets::destructible_sedan();
    for p in d.parts.iter_mut().skip(1) {
        p.mount = MountDesc { break_force: 1e12, break_torque: 1e12, anchor: None };
    }
    for w in &mut d.wheels {
        w.mount = MountDesc { break_force: 1e12, break_torque: 1e12, anchor: None };
    }
    d
}

#[test]
fn a_resting_part_mount_carries_the_part_weight() {
    let mut sim = world();
    let d = measured_sedan();
    let v = sim.add_vehicle(&d, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
    run(&mut sim, 240);
    let loads = part_loads(&sim, v);
    for (p, l) in d.parts.iter().zip(&loads).skip(1) {
        let w = p.mass * G;
        eprintln!("{:>14}: mount {:8.1} N (weight {:8.1}), moment {:8.1} N·m", p.name, l.force, w, l.torque);
        // Bodywork touches nothing at rest: its mount carries its weight.
        assert!((l.force - w).abs() < 0.05 * w + 1.0, "{}: {} vs weight {}", p.name, l.force, w);
    }
    // Each wheel mount carries the weight on that corner (tyre force minus the
    // wheel's own weight): together, the sprung weight.
    let total: f32 = wheel_loads(&sim, v).iter().map(|l| l.force).sum();
    let sprung = d.parts.iter().map(|p| p.mass).sum::<f32>() * G;
    assert!((total - sprung).abs() < 0.05 * sprung, "wheel mounts {total} vs sprung weight {sprung}");
    let st = sim.vehicle_state(v);
    assert_eq!((st.parts_broken, st.wheels_broken, st.parts_lost), (0, 0, 0));
}

#[test]
fn default_mounts_never_break() {
    // Presets without mounts are unbreakable however hard they are hit.
    let mut sim = world();
    let d = presets::destructible_sedan();
    let v = sim.add_vehicle(&d, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
    run(&mut sim, 120);
    ball(&mut sim, [0.0, 0.6, 8.0], 0.5, 800.0, [0.0, 0.0, -30.0]);
    run(&mut sim, 120);
    let st = sim.vehicle_state(v);
    assert_eq!((st.parts_broken, st.wheels_broken, st.parts_lost), (0, 0, 0));
    assert!(sim.poll_mount_breaks().is_empty());
}

#[test]
fn a_heavy_ball_snaps_the_struck_mount_only() {
    let mut sim = world();
    let mut d = measured_sedan();
    // Front bumper on a 40 kN mount; everything else out of reach.
    let bumper = d.parts.iter().position(|p| p.name == "bumper_front").unwrap();
    d.parts[bumper].mount = MountDesc { break_force: 40e3, break_torque: 40e3, anchor: None };
    let v = sim.add_vehicle(&d, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
    run(&mut sim, 180);
    assert_eq!(sim.vehicle_state(v).parts_broken, 0, "resting: nothing breaks");
    // 300 kg at 6 m/s straight into the front bumper, at its height: 0.1 m a
    // step, so the blow lands on the 0.2 m bumper, not the frame behind it.
    let at = world_point(&sim, v, d.parts[bumper].colliders[0].position);
    ball(&mut sim, [at[0], at[1], at[2] + 1.0], 0.35, 300.0, [0.0, 0.0, -6.0]);
    let mut peak = 0.0f32;
    for _ in 0..90 {
        sim.step().unwrap();
        peak = peak.max(part_loads(&sim, v)[bumper].force);
    }
    let st = sim.vehicle_state(v);
    let breaks = sim.poll_mount_breaks();
    eprintln!("bumper peak {peak:.0} N; broken {:#b}; events {breaks:?}", st.parts_broken);
    assert_eq!(st.parts_broken, 1 << bumper, "only the bumper broke");
    assert_eq!(st.parts_lost & (1 << bumper), 1 << bumper, "and left the car");
    assert_eq!(breaks.len(), 1);
    assert_eq!((breaks[0].vehicle, breaks[0].index, breaks[0].wheel), (v.0, bumper as u32, false));
    assert!(breaks[0].force > 40e3 || breaks[0].torque > 40e3);
    // The severed bumper is a free body now.
    let body = sim.vehicle_host().part_body(v, bumper as u32);
    assert!(sim.read_body(body).position[1] > -100.0, "bumper body activated");
}

/// A design-space point of `v` in the world (the chassis has not turned).
fn world_point(sim: &WgpuSim, v: VehicleHandle, p: [f32; 3]) -> [f32; 3] {
    let c = sim.read_body(sim.vehicle_host().chassis_body(v));
    let q = glam::Quat::from_array(c.orientation);
    let o = sim.vehicle_host().origin(v);
    let local = glam::Vec3::from_array(p) - glam::Vec3::from_array(o);
    (glam::Vec3::from_array(c.position) + q * local).to_array()
}

/// Peak load ratio of every mount over a run (part names, then wheels).
fn peaks(sim: &WgpuSim, v: VehicleHandle, d: &VehicleDesc) -> Vec<(String, f32)> {
    sim.vehicle_mount_loads(v)
        .iter()
        .map(|l| (if l.wheel { format!("wheel {}", l.index) } else { d.parts[l.index as usize].name.clone() }, l.peak_ratio))
        .collect()
}

#[test]
fn dune_buggy_settles_drives_and_jumps_without_shedding_parts() {
    let mut sim = world();
    let d = presets::dune_buggy();
    // A kicker ramp 25 m ahead and a landing area beyond it.
    let q = glam::Quat::from_axis_angle(glam::Vec3::X, -0.25).to_array();
    add_box(&mut sim, [0.0, 0.5, 25.0], [3.0, 0.3, 3.0], q, 0.0, [0.0; 3]);
    let v = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
    run(&mut sim, 180);
    let st = sim.vehicle_state(v);
    assert_eq!(st.contacts, 4, "all four wheels down");
    sim.set_vehicle_input(v, VehicleInput { forward: 1.0, ..Default::default() });
    let mut top = 0.0f32;
    for _ in 0..360 {
        sim.step().unwrap();
        top = top.max(sim.vehicle_state(v).forward_speed);
    }
    sim.set_vehicle_input(v, VehicleInput { brake: 1.0, ..Default::default() });
    run(&mut sim, 180);
    let st = sim.vehicle_state(v);
    for (name, r) in peaks(&sim, v, &d) {
        eprintln!("{name:>16}: peak {:.3} of its limit", r);
    }
    eprintln!("top speed {top:.1} m/s");
    assert!(top > 12.0, "drives ({top} m/s)");
    assert_eq!((st.parts_broken, st.wheels_broken), (0, 0), "a jump sheds nothing");
}

#[test]
fn a_cannonball_tears_the_struck_part_off_the_dune_buggy() {
    let mut sim = world();
    let d = presets::dune_buggy();
    let v = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
    run(&mut sim, 180);
    // The left mirror stands out from the cage (+X is left): a 100 kg ball at
    // 10 m/s (0.17 m a step: it meets the mirror before the cage) from outside.
    let mirror = d.parts.iter().position(|p| p.name == "mirror_l").unwrap();
    let housing = d.parts[mirror].colliders.iter().max_by(|a, b| a.position[0].total_cmp(&b.position[0])).unwrap().position;
    let at = world_point(&sim, v, housing);
    ball(&mut sim, [at[0] + 1.0, at[1], at[2]], 0.2, 100.0, [-10.0, 0.0, 0.0]);
    run(&mut sim, 60);
    let st = sim.vehicle_state(v);
    for (name, r) in peaks(&sim, v, &d) {
        eprintln!("{name:>16}: peak {:.3} of its limit", r);
    }
    let breaks = sim.poll_mount_breaks();
    eprintln!("broken parts {:#b} wheels {:#b}: {breaks:?}", st.parts_broken, st.wheels_broken);
    assert_eq!(st.parts_broken, 1 << mirror, "the struck mirror came off, and only it");
    assert_eq!(st.wheels_broken, 0);
    assert_eq!(breaks.len(), 1);
}

/// Calibration probe (run by hand): peak mount force of every buggy part when a
/// cannonball strikes it, and over a drive with a jump, all mounts unbreakable.
#[test]
#[ignore]
fn dune_buggy_mount_load_probe() {
    let unbreakable = || {
        let mut d = presets::dune_buggy();
        for p in d.parts.iter_mut().skip(1) {
            p.mount = MountDesc { break_force: 1e12, break_torque: 1e12, anchor: None };
        }
        for w in &mut d.wheels {
            w.mount = MountDesc { break_force: 1e12, break_torque: 1e12, anchor: None };
        }
        d
    };
    let d = unbreakable();
    let peak_force = |sim: &WgpuSim, v: VehicleHandle, peak: &mut Vec<(f32, f32)>| {
        for (k, l) in sim.vehicle_mount_loads(v).iter().enumerate() {
            peak[k].0 = peak[k].0.max(l.force);
            peak[k].1 = peak[k].1.max(l.torque);
        }
    };
    let n = d.parts.len() + d.wheels.len();
    // Drive and jump.
    let mut sim = world();
    let q = glam::Quat::from_axis_angle(glam::Vec3::X, -0.25).to_array();
    add_box(&mut sim, [0.0, 0.5, 25.0], [3.0, 0.3, 3.0], q, 0.0, [0.0; 3]);
    let v = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
    run(&mut sim, 180);
    let mut drive = vec![(0.0f32, 0.0f32); n];
    sim.set_vehicle_input(v, VehicleInput { forward: 1.0, ..Default::default() });
    for _ in 0..360 {
        sim.step().unwrap();
        peak_force(&sim, v, &mut drive);
    }
    // Cannonball at each part's first collider.
    let mut hit = vec![(0.0f32, 0.0f32); n];
    for (pi, p) in d.parts.iter().enumerate().skip(1) {
        let mut sim = world();
        let v = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
        run(&mut sim, 120);
        let c = sim.read_body(sim.vehicle_host().chassis_body(v));
        let o = sim.vehicle_host().origin(v);
        let at = p.colliders[0].position;
        let w = [c.position[0] + at[0] - o[0], c.position[1] + at[1] - o[1], c.position[2] + at[2] - o[2]];
        // From outside the car: along +X for the left side, -X right, ±Z for the ends.
        let dir = if at[2] > 1.2 { [0.0, 0.0, -1.0] } else if at[2] < -1.2 { [0.0, 0.0, 1.0] } else if at[0] >= 0.0 { [-1.0, 0.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let start = [w[0] - dir[0] * 3.0, w[1] - dir[1] * 3.0, w[2] - dir[2] * 3.0];
        ball(&mut sim, start, 0.2, 30.0, [dir[0] * 40.0, dir[1] * 40.0, dir[2] * 40.0]);
        let mut peak = vec![(0.0f32, 0.0f32); n];
        for _ in 0..30 {
            sim.step().unwrap();
            peak_force(&sim, v, &mut peak);
        }
        hit[pi] = peak[pi];
    }
    eprintln!("{:>16} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12}", "part", "drive F", "drive τ", "hit F", "hit τ", "limit F", "limit τ");
    let real = presets::dune_buggy();
    for k in 0..n {
        let (name, lim) = if k < d.parts.len() {
            (real.parts[k].name.clone(), (real.parts[k].mount.break_force, real.parts[k].mount.break_torque))
        } else {
            (format!("wheel {}", k - d.parts.len()), (real.wheels[k - d.parts.len()].mount.break_force, real.wheels[k - d.parts.len()].mount.break_torque))
        };
        eprintln!("{name:>16} {:>10.0} {:>10.0} {:>10.0} {:>10.0} {:>12.0} {:>12.0}", drive[k].0, drive[k].1, hit[k].0, hit[k].1, lim.0, lim.1);
    }
}

#[test]
#[ignore]
fn dune_buggy_hit_trace() {
    let mut d = presets::dune_buggy();
    for p in d.parts.iter_mut().skip(1) {
        p.mount = MountDesc { break_force: 1e12, break_torque: 1e12, anchor: None };
    }
    let mut sim = world();
    let v = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
    run(&mut sim, 120);
    let c = sim.read_body(sim.vehicle_host().chassis_body(v));
    let o = sim.vehicle_host().origin(v);
    let pi = std::env::var("PART").ok().and_then(|s| s.parse().ok()).unwrap_or(1usize);
    let speed: f32 = std::env::var("SPEED").ok().and_then(|s| s.parse().ok()).unwrap_or(15.0);
    let mass: f32 = std::env::var("MASS").ok().and_then(|s| s.parse().ok()).unwrap_or(300.0);
    let at = d.parts[pi].colliders[0].position;
    let w = [c.position[0] + at[0] - o[0], c.position[1] + at[1] - o[1], c.position[2] + at[2] - o[2]];
    eprintln!("part {} collider at {:?} world {:?} chassis {:?}", d.parts[pi].name, at, w, c.position);
    let b = ball(&mut sim, [w[0], w[1], w[2] + 3.0], 0.25, mass, [0.0, 0.0, -speed]);
    for i in 0..40 {
        sim.step().unwrap();
        let bb = sim.read_body(b);
        let cc = sim.read_body(sim.vehicle_host().chassis_body(v));
        let l = &sim.vehicle_mount_loads(v)[pi];
        eprintln!("{i:3} ball z {:7.3} vz {:7.2} | chassis vz {:6.2} | {} F {:9.0} τ {:7.0}", bb.position[2], bb.linear_vel[2], cc.linear_vel[2], d.parts[pi].name, l.force, l.torque);
    }
}

/// Calibration probe (run by hand): peak load ratio (load / limit) of every
/// buggy mount across crash scenarios, all mounts measured but unbreakable.
#[test]
#[ignore]
fn dune_buggy_crash_probe() {
    let real = presets::dune_buggy();
    let measured = || {
        let mut d = real.clone();
        // Limits ×1e6: ratios stay comparable (÷1e6) and nothing breaks.
        for p in d.parts.iter_mut().skip(1) {
            p.mount.break_force *= 1e6;
            p.mount.break_torque *= 1e6;
        }
        for w in &mut d.wheels {
            w.mount.break_force *= 1e6;
            w.mount.break_torque *= 1e6;
        }
        d
    };
    let d = measured();
    let names: Vec<String> = d.parts.iter().map(|p| p.name.clone()).chain((0..4).map(|i| format!("wheel {i}"))).collect();
    let mut table: Vec<(String, Vec<f32>)> = Vec::new();
    let collect = |sim: &WgpuSim, v: VehicleHandle| -> Vec<f32> { sim.vehicle_mount_loads(v).iter().map(|l| l.peak_ratio * 1e6).collect() };
    // 1. Full throttle into a concrete wall from 40 m.
    {
        let mut sim = world();
        add_box(&mut sim, [0.0, 1.0, 40.0], [6.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0], 0.0, [0.0; 3]);
        let v = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
        run(&mut sim, 60);
        sim.set_vehicle_input(v, VehicleInput { forward: 1.0, ..Default::default() });
        let mut top = 0.0f32;
        for _ in 0..360 {
            sim.step().unwrap();
            top = top.max(sim.vehicle_state(v).forward_speed);
        }
        table.push((format!("wall @{top:.0} m/s"), collect(&sim, v)));
    }
    // 2. Head-on: two buggies at full throttle from 30 m apart.
    {
        let mut sim = world();
        let a = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
        let b = sim.add_vehicle(&d, [0.2, 0.6, 50.0], glam::Quat::from_rotation_y(std::f32::consts::PI).to_array()).unwrap();
        run(&mut sim, 60);
        sim.set_vehicle_input(a, VehicleInput { forward: 1.0, ..Default::default() });
        sim.set_vehicle_input(b, VehicleInput { forward: 1.0, ..Default::default() });
        run(&mut sim, 300);
        table.push(("head-on".into(), collect(&sim, a)));
    }
    // 3. Cannonballs (400 kg, r 0.6, as vehicle.html) at 25 and 45 m/s: front and side.
    for (label, speed, from, dir) in [
        ("cannon front 25", 25.0f32, [0.0, 0.8, 8.0], [0.0, 0.0, -1.0]),
        ("cannon side 25", 25.0, [8.0, 0.9, 0.0], [-1.0, 0.0, 0.0]),
        ("cannon front 45", 45.0, [0.0, 0.8, 10.0], [0.0, 0.0, -1.0]),
        ("cannon side 45", 45.0, [10.0, 0.9, 0.0], [-1.0, 0.0, 0.0]),
        ("cannon wheel 45", 45.0, [10.0, 0.45, 1.6], [-1.0, 0.0, 0.0]),
    ] {
        let mut sim = world();
        let v = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
        run(&mut sim, 90);
        ball(&mut sim, from, 0.6, 400.0, [dir[0] * speed, dir[1] * speed, dir[2] * speed]);
        run(&mut sim, 90);
        table.push((label.into(), collect(&sim, v)));
    }
    // 4. Jump off a kicker at full throttle.
    {
        let mut sim = world();
        let q = glam::Quat::from_axis_angle(glam::Vec3::X, -0.3).to_array();
        add_box(&mut sim, [0.0, 0.55, 25.0], [3.5, 0.3, 4.0], q, 0.0, [0.0; 3]);
        let v = sim.add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
        run(&mut sim, 60);
        sim.set_vehicle_input(v, VehicleInput { forward: 1.0, ..Default::default() });
        run(&mut sim, 360);
        table.push(("jump".into(), collect(&sim, v)));
    }
    eprint!("{:>16}", "mount");
    for (label, _) in &table {
        eprint!(" {:>16}", label);
    }
    eprintln!();
    for (k, name) in names.iter().enumerate() {
        eprint!("{name:>16}");
        for (_, r) in &table {
            eprint!(" {:>16.3}", r[k]);
        }
        eprintln!();
    }
}
