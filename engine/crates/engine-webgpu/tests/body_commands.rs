//! Body commands (`push_body_commands`) give the state a host read-modify-write
//! (`read_body` → edit → `write_bodies`) gives, without the blocking readback.

mod support;

use engine_schema::{BodyState, GpuCommand, WorldCapacity};
use engine_webgpu::{ContactStorage, SimOptions, WgpuSim};
use glam::{Mat3, Quat, Vec3};

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 256,
        max_colliders: 256,
        max_contacts: 8192,
        max_joints: 16,
        max_candidates: 16_384,
        max_colors: 32,
        max_commands: 2048,
        max_hull_vertices: 64,
    }
}

const ID: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Ground plus a settled 3×3×3 pile (bodies 1..=27).
fn pile(contacts: ContactStorage) -> WgpuSim {
    let mut sim = WgpuSim::new_with_options(
        capacity(),
        SimOptions {
            contacts,
            ..SimOptions::default()
        },
    )
    .expect("wgpu");
    support::add(
        &mut sim,
        [0.0, -0.5, 0.0],
        [20.0, 0.5, 20.0],
        0.0,
        ID,
        false,
    );
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                let p = [x as f32 * 1.1, 0.5 + y as f32 * 1.02, z as f32 * 1.1];
                support::add(&mut sim, p, [0.5; 3], 1.0, ID, false);
            }
        }
    }
    for _ in 0..30 {
        sim.step().expect("step");
    }
    sim
}

fn bytes(sim: &WgpuSim) -> Vec<u8> {
    bytemuck::cast_slice::<BodyState, u8>(&sim.download_bodies()).to_vec()
}

fn wake(b: &mut BodyState) {
    if b.inv_mass > 0.0 {
        b.flags = (b.flags & !4) | 8;
        b.sleep_timer = 0.0;
    }
}

/// The host twin of each command (the read-modify-write vibe-land's bridge does).
fn host_apply(sim: &mut WgpuSim, c: &GpuCommand, dt: f32) {
    let mut b = sim.read_body(c.a);
    let v3 = |f: [f32; 4]| Vec3::new(f[0], f[1], f[2]);
    let world_inv_i = |b: &BodyState| {
        let r = Mat3::from_quat(Quat::from_array(b.orientation));
        r * Mat3::from_diagonal(b.inv_inertia.into()) * r.transpose()
    };
    match c.op {
        3 => wake(&mut b),
        4 => {
            b.position = v3(c.f0).to_array();
            b.orientation = c.f1;
            wake(&mut b);
        }
        5 => {
            b.linear_vel = v3(c.f0).to_array();
            b.angular_vel = v3(c.f1).to_array();
            wake(&mut b);
        }
        12 if b.inv_mass > 0.0 => {
            let j = v3(c.f0);
            let r = v3(c.f1) - Vec3::from(b.position);
            b.linear_vel = (Vec3::from(b.linear_vel) + j * b.inv_mass).to_array();
            b.angular_vel = (Vec3::from(b.angular_vel) + world_inv_i(&b) * r.cross(j)).to_array();
            wake(&mut b);
        }
        11 if b.inv_mass > 0.0 => {
            b.linear_vel = (Vec3::from(b.linear_vel) + v3(c.f0) * (b.inv_mass * dt)).to_array();
            b.angular_vel =
                (Vec3::from(b.angular_vel) + world_inv_i(&b) * (v3(c.f1) * dt)).to_array();
            wake(&mut b);
        }
        _ => {}
    }
    sim.write_bodies(c.a, &[b]);
}

#[test]
fn exact_commands_match_host_read_modify_write_bit_for_bit() {
    for contacts in [ContactStorage::Canonical, ContactStorage::Persistent] {
        let cmds = [
            GpuCommand::set_velocity(5, [1.0, 2.0, -0.5], [0.3, 0.0, 0.1]),
            GpuCommand::teleport(9, [4.0, 3.0, 1.0], [0.0, 0.38268343, 0.0, 0.9238795]),
            GpuCommand::wake(14),
            GpuCommand::set_velocity(20, [0.0, 5.0, 0.0], [0.0; 3]),
            GpuCommand::teleport(20, [-3.0, 2.0, 0.0], ID),
        ];
        let mut host = pile(contacts);
        let mut gpu = pile(contacts);
        assert!(
            bytes(&host) == bytes(&gpu),
            "{contacts:?}: runs diverged before commands"
        );
        for c in &cmds {
            host_apply(&mut host, c, 0.0);
        }
        gpu.push_body_commands(&cmds).expect("commands");
        for _ in 0..90 {
            host.step().expect("step");
            gpu.step().expect("step");
        }
        assert!(
            bytes(&host) == bytes(&gpu),
            "{contacts:?}: commands differ from host writes"
        );
    }
}

#[test]
fn impulse_and_force_match_the_host_math() {
    let dt = engine_schema::AvbdParams::default().dt;
    let cmds = [
        GpuCommand::apply_impulse(5, [2.0, 3.0, 0.5], [0.7, 1.4, 0.3]),
        GpuCommand::add_force(11, [40.0, 0.0, -10.0], [0.0, 3.0, 1.0]),
    ];
    let mut host = pile(ContactStorage::Canonical);
    let mut gpu = pile(ContactStorage::Canonical);
    for c in &cmds {
        host_apply(&mut host, c, dt);
    }
    gpu.push_body_commands(&cmds).expect("commands");
    host.step().expect("step");
    gpu.step().expect("step");
    for id in [5u32, 11] {
        let (h, g) = (host.read_body(id), gpu.read_body(id));
        for k in 0..3 {
            for (a, b) in [
                (h.linear_vel[k], g.linear_vel[k]),
                (h.angular_vel[k], g.angular_vel[k]),
            ] {
                assert!(
                    (a - b).abs() <= 1e-4 * a.abs().max(1.0),
                    "body {id}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn commands_on_one_body_apply_in_submission_order() {
    let run = |cmds: &[GpuCommand]| {
        let mut sim = pile(ContactStorage::Canonical);
        sim.push_body_commands(cmds).expect("commands");
        sim.step().expect("step");
        sim.read_body(13)
    };
    let set = GpuCommand::set_velocity(13, [0.0, 4.0, 0.0], [0.0; 3]);
    let kick = GpuCommand::apply_impulse(13, [3.0, 0.0, 0.0], [1.1, 1.52, 1.1]);
    let set_then_kick = run(&[set, kick]);
    let kick_then_set = run(&[kick, set]);
    assert!(
        set_then_kick.linear_vel[0] > 1.0,
        "the kick after the set survives"
    );
    assert!(
        kick_then_set.linear_vel[0].abs() < 0.5,
        "the set after the kick replaces it"
    );
    // Interleaved with other bodies' commands, order per body is kept.
    let other = GpuCommand::set_velocity(2, [0.0, 1.0, 0.0], [0.0; 3]);
    let mixed = run(&[set, other, kick, other]);
    assert!(mixed.linear_vel == set_then_kick.linear_vel);
}

#[test]
fn a_thousand_commands_on_one_body_all_apply() {
    let mut sim = pile(ContactStorage::Canonical);
    let one = GpuCommand::apply_impulse(13, [0.0, 0.01, 0.0], [1.1, 1.52, 1.1]);
    sim.push_body_commands(&vec![one; 1000]).expect("commands");
    sim.step().expect("step");
    let mut reference = pile(ContactStorage::Canonical);
    reference
        .push_body_commands(&[GpuCommand::apply_impulse(
            13,
            [0.0, 10.0, 0.0],
            [1.1, 1.52, 1.1],
        )])
        .expect("commands");
    reference.step().expect("step");
    let (a, b) = (
        sim.read_body(13).linear_vel[1],
        reference.read_body(13).linear_vel[1],
    );
    assert!((a - b).abs() < 1e-2, "{a} vs {b}");
}

#[test]
fn too_many_commands_is_an_error() {
    let mut sim = pile(ContactStorage::Canonical);
    let err = sim
        .push_body_commands(&vec![GpuCommand::wake(1); 2049])
        .err()
        .expect("past max_commands");
    assert!(err.to_string().contains("max_commands"), "{err}");
    assert!(sim.push_body_commands(&[GpuCommand::wake(999)]).is_err());
}
