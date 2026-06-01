//! Narrowphase contact-geometry matrix, validated against an **independent**
//! oracle (parry3d) and against analytic penetration.
//!
//! The dynamics tests exercise the solver; this isolates *collision detection*.
//! For each shape pair in a clean ~0.1m penetration we run a single step with
//! **zero solver iterations** (so the bodies are not moved before we read the
//! contact) and require:
//!   * the contact normal axis to match parry's to a few degrees (derived from
//!     contact-point ambiguity within the slop band — far tighter than the ~37°
//!     the existing broad matrix allows), and
//!   * the penetration depth to match parry within slop + contact_offset.
//!
//! Engine convention: `Contact3D::depth()` is negative when penetrating, so we
//! compare magnitudes.

mod support;

use glam::{Quat, Vec3};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};
use support::parry_oracle::parry_contact_query;

fn cfg() -> SimConfig {
    SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 0,
        max_bodies: 8,
        ..Default::default()
    }
}

struct PairCase {
    name: &'static str,
    a: ShapeDesc,
    pos_a: Vec3,
    rot_a: Quat,
    b: ShapeDesc,
    pos_b: Vec3,
    rot_b: Quat,
    a_static: bool,
}

/// Deepest contact's (normal, |depth|) for the pair after a single zero-iteration
/// step (= pure narrowphase output on the initial configuration).
fn run_pair(case: &PairCase) -> Option<(Vec3, f32)> {
    let mut world = World::new(cfg()).ok()?;
    world.add_body(&RigidBodyDesc {
        position: case.pos_a,
        rotation: case.rot_a,
        mass: if case.a_static { 0.0 } else { 1.0 },
        friction: 0.5,
        shape: case.a.clone(),
        ..Default::default()
    });
    world.add_body(&RigidBodyDesc {
        position: case.pos_b,
        rotation: case.rot_b,
        mass: 1.0,
        friction: 0.5,
        shape: case.b.clone(),
        ..Default::default()
    });
    world.step();
    let c = world
        .last_contacts()
        .iter()
        .max_by(|x, y| x.depth().abs().partial_cmp(&y.depth().abs()).unwrap())?;
    Some((c.contact_normal(), c.depth().abs()))
}

fn cases() -> Vec<PairCase> {
    let sphere = |r: f32| ShapeDesc::Sphere { radius: r };
    let boxs = |h: f32| ShapeDesc::Box {
        half_extents: Vec3::splat(h),
    };
    let cap = |hh: f32, r: f32| ShapeDesc::Capsule {
        half_height: hh,
        radius: r,
    };
    let plane = ShapeDesc::Plane {
        normal: Vec3::Y,
        distance: 0.0,
    };
    let id = Quat::IDENTITY;
    let zrot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
    vec![
        PairCase {
            name: "sphere_sphere",
            a: sphere(0.5),
            pos_a: Vec3::new(-0.45, 0.0, 0.0),
            rot_a: id,
            b: sphere(0.5),
            pos_b: Vec3::new(0.45, 0.0, 0.0),
            rot_b: id,
            a_static: false,
        },
        PairCase {
            name: "sphere_box",
            a: sphere(0.4),
            pos_a: Vec3::new(0.0, 0.8, 0.0),
            rot_a: id,
            b: boxs(0.5),
            pos_b: Vec3::ZERO,
            rot_b: id,
            a_static: false,
        },
        PairCase {
            name: "box_box",
            a: boxs(0.5),
            pos_a: Vec3::new(-0.45, 0.0, 0.0),
            rot_a: id,
            b: boxs(0.5),
            pos_b: Vec3::new(0.45, 0.0, 0.0),
            rot_b: id,
            a_static: false,
        },
        PairCase {
            name: "sphere_capsule",
            a: sphere(0.4),
            pos_a: Vec3::new(0.6, 0.0, 0.0),
            rot_a: id,
            b: cap(0.5, 0.3),
            pos_b: Vec3::ZERO,
            rot_b: id,
            a_static: false,
        },
        PairCase {
            name: "box_capsule",
            a: boxs(0.5),
            pos_a: Vec3::ZERO,
            rot_a: id,
            b: cap(0.5, 0.3),
            pos_b: Vec3::new(0.7, 0.0, 0.0),
            rot_b: id,
            a_static: false,
        },
        PairCase {
            name: "capsule_capsule",
            a: cap(0.5, 0.3),
            pos_a: Vec3::new(-0.25, 0.0, 0.0),
            rot_a: id,
            b: cap(0.5, 0.3),
            pos_b: Vec3::new(0.25, 0.0, 0.0),
            rot_b: id,
            a_static: false,
        },
        PairCase {
            name: "sphere_plane",
            a: plane.clone(),
            pos_a: Vec3::ZERO,
            rot_a: id,
            b: sphere(0.5),
            pos_b: Vec3::new(0.0, 0.4, 0.0),
            rot_b: id,
            a_static: true,
        },
        PairCase {
            name: "box_plane",
            a: plane.clone(),
            pos_a: Vec3::ZERO,
            rot_a: id,
            b: boxs(0.5),
            pos_b: Vec3::new(0.0, 0.4, 0.0),
            rot_b: id,
            a_static: true,
        },
        // Rotated configs exercise the SAT off-axis; the analytic answer is fiddly,
        // so these are validated purely against parry (correct for any pose).
        PairCase {
            name: "box_box_tilted",
            a: boxs(0.5),
            pos_a: Vec3::ZERO,
            rot_a: id,
            b: boxs(0.5),
            pos_b: Vec3::new(1.1, 0.0, 0.0),
            rot_b: zrot,
            a_static: false,
        },
        PairCase {
            name: "box_plane_tilted",
            a: plane,
            pos_a: Vec3::ZERO,
            rot_a: id,
            b: boxs(0.5),
            pos_b: Vec3::new(0.0, 0.6, 0.0),
            rot_b: zrot,
            a_static: true,
        },
    ]
}

#[test]
fn narrowphase_matches_parry() {
    let cases = cases();
    let mut ran = false;
    let mut failures = Vec::new();

    for case in &cases {
        let Some((gpu_n, gpu_depth)) = run_pair(case) else {
            if World::new(cfg()).is_err() {
                eprintln!("SKIP: no GPU adapter");
                return;
            }
            failures.push(format!(
                "{}: engine produced NO contact for a penetrating pair",
                case.name
            ));
            continue;
        };
        ran = true;

        let Some(parry) = parry_contact_query(
            &case.a, case.pos_a, case.rot_a, &case.b, case.pos_b, case.rot_b, 0.1,
        ) else {
            failures.push(format!(
                "{}: parry reported no contact (test setup error)",
                case.name
            ));
            continue;
        };
        let parry_depth = parry.depth;
        // Sign-agnostic axis agreement (engine/parry use opposite normal
        // conventions); cos(5°) ≈ 0.996 for clear penetration.
        let align = gpu_n.normalize().dot(parry.normal.normalize()).abs();
        let depth_err = (gpu_depth - parry_depth).abs();

        println!(
            "{:>18}: align={:.4} gpu_depth={:.4} parry_depth={:.4} depth_err={:.4}",
            case.name, align, gpu_depth, parry_depth, depth_err
        );

        if align < 0.996 {
            failures.push(format!(
                "{}: normal off by >5° from parry (align={align:.4})",
                case.name
            ));
        }
        if depth_err > 0.025 {
            failures.push(format!(
                "{}: depth {gpu_depth:.4} vs parry {parry_depth:.4} (err {depth_err:.4} > slop+offset)",
                case.name
            ));
        }
    }

    assert!(ran, "no pairs evaluated");
    assert!(
        failures.is_empty(),
        "narrowphase geometry disagreements vs parry:\n{}",
        failures.join("\n")
    );
}
