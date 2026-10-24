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

fn add_box(
    sim: &mut WgpuSim,
    pos: [f32; 3],
    half: [f32; 3],
    rot: [f32; 4],
    inv_mass: f32,
    inv_i: [f32; 3],
) -> u32 {
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
        params: [
            half[0],
            half[1],
            half[2],
            (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
        ],
    });
    body
}

/// A free sphere of `mass` at `pos` moving at `vel`.
fn ball(sim: &mut WgpuSim, pos: [f32; 3], radius: f32, mass: f32, vel: [f32; 3]) -> u32 {
    let i = 0.4 * mass * radius * radius;
    let mut b = BodyState::dynamic(
        glam::Vec3::from_array(pos),
        glam::Quat::IDENTITY,
        mass,
        glam::Vec3::splat(i),
    );
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
    world_on(WgpuSim::new(capacity()).expect("wgpu"))
}

/// Contact storage variants: vehicle_loads reads contacts as each one stores them.
#[derive(Clone, Copy, Debug)]
enum Storage {
    Canonical,
    Compact,
    Persistent,
}

fn world_with(storage: Storage) -> WgpuSim {
    world_on(
        match storage {
            Storage::Canonical => WgpuSim::new_with_compact_contacts(capacity(), false),
            Storage::Compact => WgpuSim::new_with_compact_contacts(capacity(), true),
            Storage::Persistent => WgpuSim::new_with_persistent_contacts(capacity()),
        }
        .expect("wgpu"),
    )
}

fn world_on(mut sim: WgpuSim) -> WgpuSim {
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(
        0,
        Material {
            friction: 1.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    add_box(
        &mut sim,
        [0.0, -5.0, 0.0],
        [400.0, 5.0, 400.0],
        [0.0, 0.0, 0.0, 1.0],
        0.0,
        [0.0; 3],
    );
    sim
}

fn run(sim: &mut WgpuSim, steps: u32) {
    for _ in 0..steps {
        sim.step().expect("step");
    }
}

fn part_loads(sim: &WgpuSim, v: VehicleHandle) -> Vec<MountLoad> {
    sim.vehicle_mount_loads(v)
        .into_iter()
        .filter(|l| !l.wheel)
        .collect()
}

fn wheel_loads(sim: &WgpuSim, v: VehicleHandle) -> Vec<MountLoad> {
    sim.vehicle_mount_loads(v)
        .into_iter()
        .filter(|l| l.wheel)
        .collect()
}

/// The destructible sedan with every mount finite but far out of reach, so the
/// loads are measured and nothing breaks.
fn measured_sedan() -> VehicleDesc {
    let mut d = presets::destructible_sedan();
    for p in d.parts.iter_mut().skip(1) {
        p.mount = MountDesc {
            break_force: 1e12,
            break_torque: 1e12,
            anchor: None,
            parent: None,
        };
    }
    for w in &mut d.wheels {
        w.mount = MountDesc {
            break_force: 1e12,
            break_torque: 1e12,
            anchor: None,
            parent: None,
        };
    }
    d
}

#[test]
fn a_resting_part_mount_carries_the_part_weight() {
    let mut sim = world();
    let d = measured_sedan();
    let v = sim
        .add_vehicle(&d, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    run(&mut sim, 240);
    let loads = part_loads(&sim, v);
    for (p, l) in d.parts.iter().zip(&loads).skip(1) {
        let w = p.mass * G;
        eprintln!(
            "{:>14}: mount {:8.1} N (weight {:8.1}), moment {:8.1} N·m",
            p.name, l.force, w, l.torque
        );
        // Bodywork touches nothing at rest: its mount carries its weight.
        assert!(
            (l.force - w).abs() < 0.05 * w + 1.0,
            "{}: {} vs weight {}",
            p.name,
            l.force,
            w
        );
    }
    // Each wheel mount carries the weight on that corner (tyre force minus the
    // wheel's own weight): together, the sprung weight.
    let total: f32 = wheel_loads(&sim, v).iter().map(|l| l.force).sum();
    let sprung = d.parts.iter().map(|p| p.mass).sum::<f32>() * G;
    assert!(
        (total - sprung).abs() < 0.05 * sprung,
        "wheel mounts {total} vs sprung weight {sprung}"
    );
    let st = sim.vehicle_state(v);
    assert_eq!(
        (st.parts_broken, st.wheels_broken, st.parts_lost),
        (0, 0, 0)
    );
}

#[test]
fn default_mounts_never_break() {
    // Presets without mounts are unbreakable however hard they are hit.
    let mut sim = world();
    let d = presets::destructible_sedan();
    let v = sim
        .add_vehicle(&d, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    run(&mut sim, 120);
    ball(&mut sim, [0.0, 0.6, 8.0], 0.5, 800.0, [0.0, 0.0, -30.0]);
    run(&mut sim, 120);
    let st = sim.vehicle_state(v);
    assert_eq!(
        (st.parts_broken, st.wheels_broken, st.parts_lost),
        (0, 0, 0)
    );
    assert!(sim.poll_mount_breaks().is_empty());
}

#[test]
fn a_heavy_ball_snaps_the_struck_mount_only() {
    for storage in [Storage::Canonical, Storage::Compact, Storage::Persistent] {
        eprintln!("--- {storage:?} contacts");
        heavy_ball_snaps_the_struck_mount_only(storage);
    }
}

fn heavy_ball_snaps_the_struck_mount_only(storage: Storage) {
    let mut sim = world_with(storage);
    let mut d = measured_sedan();
    // Front bumper on a 40 kN mount; everything else out of reach.
    let bumper = d
        .parts
        .iter()
        .position(|p| p.name == "bumper_front")
        .unwrap();
    d.parts[bumper].mount = MountDesc {
        break_force: 40e3,
        break_torque: 40e3,
        anchor: None,
        parent: None,
    };
    let v = sim
        .add_vehicle(&d, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    run(&mut sim, 180);
    assert_eq!(
        sim.vehicle_state(v).parts_broken,
        0,
        "resting: nothing breaks"
    );
    // 300 kg at 6 m/s straight into the front bumper, at its height: 0.1 m a
    // step, so the blow lands on the 0.2 m bumper, not the frame behind it.
    let at = world_point(&sim, v, d.parts[bumper].colliders[0].position);
    ball(
        &mut sim,
        [at[0], at[1], at[2] + 1.0],
        0.35,
        300.0,
        [0.0, 0.0, -6.0],
    );
    let mut peak = 0.0f32;
    for _ in 0..90 {
        sim.step().unwrap();
        peak = peak.max(part_loads(&sim, v)[bumper].force);
    }
    let st = sim.vehicle_state(v);
    let breaks = sim.poll_mount_breaks();
    eprintln!(
        "bumper peak {peak:.0} N; broken {:#b}; events {breaks:?}",
        st.parts_broken
    );
    assert_eq!(st.parts_broken, 1 << bumper, "only the bumper broke");
    assert_eq!(
        st.parts_lost & (1 << bumper),
        1 << bumper,
        "and left the car"
    );
    assert_eq!(breaks.len(), 1);
    assert_eq!(
        (breaks[0].vehicle, breaks[0].index, breaks[0].wheel),
        (v.0, bumper as u32, false)
    );
    assert!(breaks[0].force > 40e3 || breaks[0].torque > 40e3);
    // The severed bumper is a free body now.
    let body = sim.vehicle_host().part_body(v, bumper as u32);
    assert!(
        sim.read_body(body).position[1] > -100.0,
        "bumper body activated"
    );
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
        .map(|l| {
            (
                if l.wheel {
                    format!("wheel {}", l.index)
                } else {
                    d.parts[l.index as usize].name.clone()
                },
                l.peak_ratio,
            )
        })
        .collect()
}

#[test]
fn dune_buggy_settles_drives_and_jumps_without_shedding_parts() {
    let mut sim = world();
    let d = presets::dune_buggy();
    // A kicker ramp 25 m ahead and a landing area beyond it.
    let q = glam::Quat::from_axis_angle(glam::Vec3::X, -0.25).to_array();
    add_box(
        &mut sim,
        [0.0, 0.5, 25.0],
        [3.0, 0.3, 3.0],
        q,
        0.0,
        [0.0; 3],
    );
    let v = sim
        .add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    run(&mut sim, 180);
    let st = sim.vehicle_state(v);
    assert_eq!(st.contacts, 4, "all four wheels down");
    sim.set_vehicle_input(
        v,
        VehicleInput {
            forward: 1.0,
            ..Default::default()
        },
    );
    let mut top = 0.0f32;
    for _ in 0..360 {
        sim.step().unwrap();
        top = top.max(sim.vehicle_state(v).forward_speed);
    }
    sim.set_vehicle_input(
        v,
        VehicleInput {
            brake: 1.0,
            ..Default::default()
        },
    );
    run(&mut sim, 180);
    let st = sim.vehicle_state(v);
    for (name, r) in peaks(&sim, v, &d) {
        eprintln!("{name:>16}: peak {:.3} of its limit", r);
    }
    eprintln!("top speed {top:.1} m/s");
    assert!(top > 12.0, "drives ({top} m/s)");
    assert_eq!(
        (st.parts_broken, st.wheels_broken),
        (0, 0),
        "a jump sheds nothing"
    );
}

#[test]
fn every_vibe_land_model_drives_and_jumps_without_shedding_parts() {
    for d in [
        presets::vibe_trophy(),
        presets::vibe_rally(),
        presets::vibe_monster(),
        presets::vibe_derby(),
        presets::vibe_sprint(),
        presets::vibe_semi(),
    ] {
        let mut sim = world();
        let q = glam::Quat::from_axis_angle(glam::Vec3::X, -0.25).to_array();
        add_box(
            &mut sim,
            [0.0, 0.5, 25.0],
            [3.0, 0.3, 3.0],
            q,
            0.0,
            [0.0; 3],
        );
        let lift = if d.name == "vibe_monster" { 1.0 } else { 0.6 };
        let v = sim
            .add_vehicle(&d, [0.0, lift, 0.0], [0.0, 0.0, 0.0, 1.0])
            .unwrap();
        run(&mut sim, 180);
        sim.set_vehicle_input(
            v,
            VehicleInput {
                forward: 1.0,
                ..Default::default()
            },
        );
        run(&mut sim, 360);
        sim.set_vehicle_input(
            v,
            VehicleInput {
                brake: 1.0,
                ..Default::default()
            },
        );
        run(&mut sim, 180);
        let st = sim.vehicle_state(v);
        let worst = peaks(&sim, v, &d)
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        eprintln!(
            "{:>13}: broken parts {:#b} wheels {:#b}; heaviest mount {} at {:.2} of its limit",
            d.name, st.parts_broken, st.wheels_broken, worst.0, worst.1
        );
        // The recipe hangs these by one joint (a cantilever): the sprint's nerf
        // bars meet the frame only at the front (their rear return stops short of
        // the rail), the semi's stacks each stand on one bracket. A hard landing
        // can bend them off; nothing else may come off.
        let fragile: &[&str] = match d.name.as_str() {
            "vibe_sprint" => &["nerf_l", "nerf_r"],
            "vibe_semi" => &["stack_l", "stack_r"],
            _ => &[],
        };
        let allowed = d
            .parts
            .iter()
            .enumerate()
            .filter(|(_, p)| fragile.contains(&p.name.as_str()))
            .fold(0u32, |m, (i, _)| m | (1 << i));
        assert_eq!(
            st.parts_broken & !allowed,
            0,
            "{}: a jump sheds nothing else",
            d.name
        );
        assert_eq!(st.wheels_broken, 0, "{}: no wheel comes off", d.name);
    }
}

#[test]
fn a_cannonball_tears_the_struck_part_off_the_dune_buggy() {
    let mut sim = world();
    let d = presets::dune_buggy();
    let v = sim
        .add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    run(&mut sim, 180);
    // The left mirror stands out from the cage (+X is left): a 100 kg ball at
    // 10 m/s (0.17 m a step: it meets the mirror before the cage) from outside.
    let mirror = d.parts.iter().position(|p| p.name == "mirror_l").unwrap();
    let housing = d.parts[mirror]
        .colliders
        .iter()
        .max_by(|a, b| a.position[0].total_cmp(&b.position[0]))
        .unwrap()
        .position;
    let at = world_point(&sim, v, housing);
    ball(
        &mut sim,
        [at[0] + 1.0, at[1], at[2]],
        0.2,
        100.0,
        [-10.0, 0.0, 0.0],
    );
    run(&mut sim, 60);
    let st = sim.vehicle_state(v);
    for (name, r) in peaks(&sim, v, &d) {
        eprintln!("{name:>16}: peak {:.3} of its limit", r);
    }
    let breaks = sim.poll_mount_breaks();
    eprintln!(
        "broken parts {:#b} wheels {:#b}: {breaks:?}",
        st.parts_broken, st.wheels_broken
    );
    assert_eq!(
        st.parts_broken,
        1 << mirror,
        "the struck mirror came off, and only it"
    );
    assert_eq!(st.wheels_broken, 0);
    assert_eq!(breaks.len(), 1);
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
    let names: Vec<String> = d
        .parts
        .iter()
        .map(|p| p.name.clone())
        .chain((0..4).map(|i| format!("wheel {i}")))
        .collect();
    let mut table: Vec<(String, Vec<f32>)> = Vec::new();
    let collect = |sim: &WgpuSim, v: VehicleHandle| -> Vec<f32> {
        sim.vehicle_mount_loads(v)
            .iter()
            .map(|l| l.peak_ratio * 1e6)
            .collect()
    };
    // 1. Full throttle into a concrete wall from 40 m.
    {
        let mut sim = world();
        add_box(
            &mut sim,
            [0.0, 1.0, 40.0],
            [6.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
            0.0,
            [0.0; 3],
        );
        let v = sim
            .add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0])
            .unwrap();
        run(&mut sim, 60);
        sim.set_vehicle_input(
            v,
            VehicleInput {
                forward: 1.0,
                ..Default::default()
            },
        );
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
        let a = sim
            .add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let b = sim
            .add_vehicle(
                &d,
                [0.2, 0.6, 50.0],
                glam::Quat::from_rotation_y(std::f32::consts::PI).to_array(),
            )
            .unwrap();
        run(&mut sim, 60);
        sim.set_vehicle_input(
            a,
            VehicleInput {
                forward: 1.0,
                ..Default::default()
            },
        );
        sim.set_vehicle_input(
            b,
            VehicleInput {
                forward: 1.0,
                ..Default::default()
            },
        );
        run(&mut sim, 300);
        table.push(("head-on".into(), collect(&sim, a)));
    }
    // 3. Cannonballs (400 kg, r 0.6, as vehicle.html) at 25 and 45 m/s: front and side.
    for (label, speed, from, dir) in [
        (
            "cannon front 25",
            25.0f32,
            [0.0, 0.8, 8.0],
            [0.0, 0.0, -1.0],
        ),
        ("cannon side 25", 25.0, [8.0, 0.9, 0.0], [-1.0, 0.0, 0.0]),
        ("cannon front 45", 45.0, [0.0, 0.8, 10.0], [0.0, 0.0, -1.0]),
        ("cannon side 45", 45.0, [10.0, 0.9, 0.0], [-1.0, 0.0, 0.0]),
        ("cannon wheel 45", 45.0, [10.0, 0.45, 1.6], [-1.0, 0.0, 0.0]),
    ] {
        let mut sim = world();
        let v = sim
            .add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0])
            .unwrap();
        run(&mut sim, 90);
        ball(
            &mut sim,
            from,
            0.6,
            400.0,
            [dir[0] * speed, dir[1] * speed, dir[2] * speed],
        );
        run(&mut sim, 90);
        table.push((label.into(), collect(&sim, v)));
    }
    // 4. Jump off a kicker at full throttle.
    {
        let mut sim = world();
        let q = glam::Quat::from_axis_angle(glam::Vec3::X, -0.3).to_array();
        add_box(
            &mut sim,
            [0.0, 0.55, 25.0],
            [3.5, 0.3, 4.0],
            q,
            0.0,
            [0.0; 3],
        );
        let v = sim
            .add_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0])
            .unwrap();
        run(&mut sim, 60);
        sim.set_vehicle_input(
            v,
            VehicleInput {
                forward: 1.0,
                ..Default::default()
            },
        );
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

/// The destructible sedan with its front bumper bolted to the hood (not the
/// chassis), every mount measured but out of reach.
fn sedan_with_bumper_on_hood() -> (VehicleDesc, usize, usize) {
    let mut d = measured_sedan();
    let hood = d.parts.iter().position(|p| p.name == "hood").unwrap();
    let bumper = d
        .parts
        .iter()
        .position(|p| p.name == "bumper_front")
        .unwrap();
    assert!(hood < bumper);
    d.parts[bumper].mount.parent = Some(hood as u32);
    (d, hood, bumper)
}

#[test]
fn a_child_part_loads_its_parents_mount() {
    let mut sim = world();
    let (d, hood, bumper) = sedan_with_bumper_on_hood();
    let v = sim
        .add_vehicle(&d, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    run(&mut sim, 240);
    let loads = part_loads(&sim, v);
    let (wh, wb) = (d.parts[hood].mass * G, d.parts[bumper].mass * G);
    eprintln!(
        "hood mount {:.0} N (hood {wh:.0} + bumper {wb:.0}); bumper mount {:.0} N",
        loads[hood].force, loads[bumper].force
    );
    // The bumper's mount carries its own weight, the hood's carries both.
    assert!((loads[bumper].force - wb).abs() < 0.05 * wb + 1.0);
    assert!((loads[hood].force - (wh + wb)).abs() < 0.05 * (wh + wb) + 1.0);
}

#[test]
fn a_severed_parent_takes_its_children_with_it() {
    let mut sim = world();
    let (d, hood, bumper) = sedan_with_bumper_on_hood();
    let v = sim
        .add_vehicle(&d, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    run(&mut sim, 120);
    let mass0 = sim.vehicle_state(v).mass;
    sim.detach_part(v, hood as u32);
    run(&mut sim, 60);
    let st = sim.vehicle_state(v);
    let lost = (1 << hood) | (1 << bumper);
    assert_eq!(
        st.parts_lost & lost,
        lost,
        "hood and the bumper bolted to it left the car"
    );
    let (mh, mb) = (d.parts[hood].mass, d.parts[bumper].mass);
    assert!(
        (mass0 - st.mass - (mh + mb)).abs() < 1.0,
        "chassis lost both masses: {mass0} → {}",
        st.mass
    );
    // One piece: the hood's body carries both masses; the bumper's own stays parked.
    let piece = sim.read_body(sim.vehicle_host().part_body(v, hood as u32));
    assert!(
        (1.0 / piece.inv_mass - (mh + mb)).abs() < 0.5,
        "piece mass {}",
        1.0 / piece.inv_mass
    );
    let parked = sim.read_body(sim.vehicle_host().part_body(v, bumper as u32));
    assert!(
        parked.position[1] < -1000.0,
        "the bumper's own body was never used"
    );
    assert!(piece.position[1] > -100.0 && piece.position.iter().all(|x| x.is_finite()));
}
