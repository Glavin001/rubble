//! D12: GPU↔CPU parity. Replays one full engine step of the simplest contact
//! scenario (sphere resting into a static plane) through the CPU mirror of the
//! solver math and requires the GPU result to match within f32 slack.
//!
//! This pins the *entire* solve chain — predict, narrowphase anchor generation,
//! contact prep (Taylor cache + penalty floor), N×(primal+dual), velocity
//! extraction — for the one-contact case. Any shader math change that isn't
//! mirrored in `rubble3d::mirror` fails here.

use glam::{Mat3, Quat, Vec3};
use rubble3d::mirror::{
    dual_update, primal_body_update, MirrorBody, MirrorContact, MirrorParams, CONTACT_MARGIN,
};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

#[test]
fn gpu_matches_cpu_mirror_sphere_on_plane() {
    let config = SimConfig::default();
    let dt = config.dt;
    let radius = 0.5f32;
    let start_y = 0.49f32; // 0.01 into the plane: one persistent contact from step 1

    let mut world = match World::new(config.clone()) {
        Ok(w) => w,
        Err(_) => {
            eprintln!("SKIP: No GPU adapter found");
            return;
        }
    };
    let plane = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });
    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, start_y, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius },
        ..Default::default()
    });
    let _ = plane;

    world.step();
    let gpu_pos = world.get_position(sphere).unwrap();
    let gpu_vel = world.get_velocity(sphere).unwrap();

    // ---------------- CPU mirror of the same step ----------------
    let params = MirrorParams {
        dt,
        beta: config.beta,
        k_max_normal: config.k_max_normal,
        k_max_tangential: config.k_max_tangential,
        alpha_stab: config.contact_stabilization,
        dynamic_mu_scale: config.dynamic_friction_ratio,
    };
    let gravity = config.gravity;
    let inv_inertia = Mat3::IDENTITY * (1.0 / (0.4 * radius * radius)); // solid sphere, m=1

    // Predict: first step has prev_vel == vel, so accel_weight = 0 and the
    // warm-start position is pos + v*dt (v = 0). The inertial target adds g*dt².
    let pos0 = Vec3::new(0.0, start_y, 0.0);
    let warm_pos = pos0;
    let inertial_pos = pos0 + gravity * (dt * dt);

    // Bodies in compacted slot order (plane = 0, sphere = 1), at solve start.
    let mk = |pos: Vec3, inv_mass: f32| MirrorBody {
        position: pos,
        orientation: Quat::IDENTITY,
        inv_mass,
        inv_inertia: if inv_mass > 0.0 {
            inv_inertia
        } else {
            Mat3::ZERO
        },
        friction: 0.5,
    };
    let solve_start = vec![mk(Vec3::ZERO, 0.0), mk(warm_pos, 1.0)];
    let inertial = vec![mk(Vec3::ZERO, 0.0), mk(inertial_pos, 1.0)];
    let mut live = solve_start.clone();

    // Narrowphase (plane_sphere_test at the solve-start pose):
    //   d = n·p − dist; depth = d − r; plane_point = p − n·d
    //   world_a = plane_point + n·depth (on the sphere), world_b = plane_point
    let d = warm_pos.y;
    let depth = d - radius;
    assert!(depth < 0.0, "test setup must start in contact");
    let plane_point = warm_pos - Vec3::Y * d;
    let world_a = plane_point + Vec3::Y * depth;
    let world_b = plane_point;
    let local_a = world_a - warm_pos; // identity orientation
    let local_b = world_b; // plane at origin, identity

    // Contact prep: C0 from the solve-start anchors, penalty floor m/dt².
    let c0_n = Vec3::Y.dot(world_a - world_b);
    let k_floor = 1.0 / (dt * dt);
    let k0 = config.k_start.max(k_floor).min(config.k_max_normal);
    let k0_t = config
        .k_start
        .max(k_floor.min(config.k_max_tangential))
        .min(config.k_max_tangential);
    let mut contact = MirrorContact {
        normal: Vec3::Y,
        tangent1: {
            // WGSL build_tangent for n = +Y: axis = +Z (|n.z| <= 0.707),
            // t = normalize(cross(axis, n)).
            Vec3::Z.cross(Vec3::Y).normalize()
        },
        c0: Vec3::new(c0_n, 0.0, 0.0),
        local_anchor_a: local_b, // slot order: contact body_a = sphere (slot 1)? see below
        local_anchor_b: local_b,
        lambda: Vec3::ZERO,
        penalty: Vec3::new(k0, k0_t, k0_t),
        body_a: 1,
        body_b: 0,
    };
    // emit_plane_contact(body_dynamic, body_plane): body_a = sphere slot (1),
    // body_b = plane slot (0). Anchors: a on the sphere, b on the plane.
    contact.local_anchor_a = local_a;
    contact.local_anchor_b = local_b;

    // N iterations of primal (sphere only; static plane early-outs) + dual.
    let mut contacts = vec![contact];
    for _ in 0..config.solver_iterations {
        let (p, q) = primal_body_update(1, &live, &inertial, &solve_start, &contacts, &params);
        live[1].position = p;
        live[1].orientation = q;
        let (next, _) = dual_update(&contacts[0], &live, &solve_start, &params);
        contacts[0] = next;
    }

    // Extract: v = (pos_final − pos_before_predict)/dt (old_states snapshot).
    let mirror_vel = (live[1].position - pos0) / dt;
    let mirror_pos = live[1].position;

    let pos_err = (gpu_pos - mirror_pos).length();
    let vel_err = (gpu_vel - mirror_vel).length();
    assert!(
        pos_err < 5e-4,
        "GPU/CPU position mismatch: gpu={gpu_pos:?} mirror={mirror_pos:?} err={pos_err}"
    );
    assert!(
        vel_err < 5e-2,
        "GPU/CPU velocity mismatch: gpu={gpu_vel:?} mirror={mirror_vel:?} err={vel_err}"
    );
    // And the physical outcome: the contact holds the sphere at/above the
    // surface minus slop, instead of letting it sink.
    assert!(
        mirror_pos.y > start_y - 0.01,
        "contact should hold the sphere: {mirror_pos:?}"
    );
    let _ = CONTACT_MARGIN;
}
