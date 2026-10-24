//! The C ABI end to end, called the way a C host calls it.

use rubble_ffi::coords::RbCoordinates;
use rubble_ffi::*;
use std::ffi::CStr;
use std::ptr;

fn ok(r: RbResult) {
    if r != RbResult::RB_OK {
        let msg = unsafe { CStr::from_ptr(rbGetLastError()) };
        panic!("{r:?}: {}", msg.to_string_lossy());
    }
}

unsafe fn world(coordinates: RbCoordinates) -> *mut RbWorld {
    let mut def = rbDefaultWorldDef();
    def.coordinates = coordinates;
    if coordinates == RbCoordinates::RB_COORDINATES_UNREAL {
        def.gravity = RbVec3 {
            x: 0.0,
            y: 0.0,
            z: -981.0,
        };
    }
    let mut w = ptr::null_mut();
    ok(rbCreateWorld(&def, &mut w));
    w
}

unsafe fn ground(w: *mut RbWorld, unreal: bool) -> RbBodyId {
    let mut def = rbDefaultBodyDef();
    let (pos, half) = if unreal {
        (
            RbVec3 {
                x: 0.0,
                y: 0.0,
                z: -50.0,
            },
            RbVec3 {
                x: 4000.0,
                y: 4000.0,
                z: 50.0,
            },
        )
    } else {
        (
            RbVec3 {
                x: 0.0,
                y: -0.5,
                z: 0.0,
            },
            RbVec3 {
                x: 40.0,
                y: 0.5,
                z: 40.0,
            },
        )
    };
    def.transform.p = pos;
    let mut g = 0;
    ok(rbCreateBody(w, &def, &mut g));
    let mut s = 0;
    ok(rbCreateBoxShape(
        w,
        g,
        &rbDefaultShapeDef(),
        half,
        RbTransform::default(),
        &mut s,
    ));
    g
}

unsafe fn poll_all(w: *mut RbWorld) -> RbFrame {
    let mut f = std::mem::zeroed();
    ok(rbWorld_Poll(w, true, true, &mut f));
    f
}

#[test]
fn a_box_falls_rests_and_sleeps_through_the_abi() {
    unsafe {
        let w = world(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        ground(w, false);
        let mut def = rbDefaultBodyDef();
        def.type_ = RbBodyType::RB_DYNAMIC_BODY;
        def.transform.p = RbVec3 {
            x: 0.0,
            y: 3.0,
            z: 0.0,
        };
        def.user_data = 77;
        let mut b = 0;
        ok(rbCreateBody(w, &def, &mut b));
        let mut s = 0;
        ok(rbCreateBoxShape(
            w,
            b,
            &rbDefaultShapeDef(),
            RbVec3 {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
            RbTransform::default(),
            &mut s,
        ));
        let mut slept = false;
        for _ in 0..400 {
            ok(rbWorld_Step(w));
            let f = poll_all(w);
            if f.has_frame {
                let moves = std::slice::from_raw_parts(f.moves, f.move_count as usize);
                slept |= moves
                    .iter()
                    .any(|m| m.body == b && m.fell_asleep && m.user_data == 77);
            }
        }
        assert!(slept, "the box fell asleep");
        let mut t = RbTransform::default();
        ok(rbBody_GetTransform(w, b, &mut t));
        assert!((t.p.y - 0.5).abs() < 0.05, "rests on the ground: {t:?}");
        // The flat-array read agrees.
        let mut flat = [RbTransform::default()];
        ok(rbWorld_GetTransforms(w, &b, 1, flat.as_mut_ptr()));
        assert_eq!(flat[0], t);
        ok(rbDestroyBody(w, b));
        assert!(!rbBody_IsValid(w, b));
        assert_eq!(rbBody_SetAwake(w, b, true), RbResult::RB_STALE_HANDLE);
        rbDestroyWorld(w);
    }
}

#[test]
fn unreal_coordinates_fall_along_minus_z_in_centimetres() {
    unsafe {
        let w = world(RbCoordinates::RB_COORDINATES_UNREAL);
        ground(w, true);
        let mut def = rbDefaultBodyDef();
        def.type_ = RbBodyType::RB_DYNAMIC_BODY;
        def.transform.p = RbVec3 {
            x: 120.0,
            y: -30.0,
            z: 300.0,
        };
        let mut b = 0;
        ok(rbCreateBody(w, &def, &mut b));
        let mut sd = rbDefaultShapeDef();
        sd.density = 0.001; // kg/cm³ (1000 kg/m³)
        let mut s = 0;
        ok(rbCreateSphereShape(
            w,
            b,
            &sd,
            RbVec3::default(),
            50.0,
            &mut s,
        ));
        // A sideways push along Unreal +Y (right).
        ok(rbBody_SetLinearVelocity(
            w,
            b,
            RbVec3 {
                x: 0.0,
                y: 100.0,
                z: 0.0,
            },
        ));
        for _ in 0..30 {
            ok(rbWorld_Step(w));
        }
        poll_all(w);
        let mut t = RbTransform::default();
        ok(rbBody_GetTransform(w, b, &mut t));
        assert!(t.p.z < 300.0, "falls down Unreal Z: {t:?}");
        assert!(t.p.y > -30.0 + 20.0, "moved right along +Y: {t:?}");
        assert!((t.p.x - 120.0).abs() < 1.0, "no drift along X: {t:?}");
        rbDestroyWorld(w);
    }
}

#[test]
fn definitions_must_come_from_their_default_functions() {
    unsafe {
        let w = world(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        let mut def = rbDefaultBodyDef();
        def.internal_value = 0;
        let mut b = 0;
        assert_eq!(rbCreateBody(w, &def, &mut b), RbResult::RB_INVALID_ARGUMENT);
        assert_eq!(
            rbCreateBody(w, ptr::null(), &mut b),
            RbResult::RB_INVALID_ARGUMENT
        );
        let mut kinematic = rbDefaultBodyDef();
        kinematic.type_ = RbBodyType::RB_KINEMATIC_BODY;
        assert_eq!(
            rbCreateBody(w, &kinematic, &mut b),
            RbResult::RB_UNSUPPORTED
        );
        let msg = CStr::from_ptr(rbGetLastError()).to_string_lossy();
        assert!(msg.contains("kinematic"), "{msg}");
        rbDestroyWorld(w);
    }
}

#[test]
fn breakable_joints_report_through_the_frame() {
    unsafe {
        let w = world(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        let mut a_def = rbDefaultBodyDef();
        a_def.transform.p = RbVec3 {
            x: 0.0,
            y: 10.0,
            z: 0.0,
        };
        let mut a = 0;
        ok(rbCreateBody(w, &a_def, &mut a));
        let mut b_def = rbDefaultBodyDef();
        b_def.type_ = RbBodyType::RB_DYNAMIC_BODY;
        b_def.transform.p = RbVec3 {
            x: 0.0,
            y: 9.0,
            z: 0.0,
        };
        let mut b = 0;
        ok(rbCreateBody(w, &b_def, &mut b));
        let mut sd = rbDefaultShapeDef();
        sd.density = 1000.0;
        let mut s = 0;
        ok(rbCreateBoxShape(
            w,
            b,
            &sd,
            RbVec3 {
                x: 0.25,
                y: 0.25,
                z: 0.25,
            },
            RbTransform::default(),
            &mut s,
        ));
        let mut j = rbDefaultSphericalJointDef();
        j.base.body_a = a;
        j.base.body_b = b;
        j.base.local_frame_a.p = RbVec3 {
            x: 0.0,
            y: -0.5,
            z: 0.0,
        };
        j.base.local_frame_b.p = RbVec3 {
            x: 0.0,
            y: 0.5,
            z: 0.0,
        };
        j.base.break_force = 100.0;
        j.base.user_data = 5;
        let mut joint = 0;
        ok(rbCreateSphericalJoint(w, &j, &mut joint));
        let mut broke = false;
        for _ in 0..30 {
            ok(rbWorld_Step(w));
            let f = poll_all(w);
            if f.has_frame {
                let events =
                    std::slice::from_raw_parts(f.destruction, f.destruction_count as usize);
                broke |= events.iter().any(|e| {
                    e.type_ == RbDestructionType::RB_JOINT_BROKE
                        && e.joint == joint
                        && e.user_data == 5
                });
            }
        }
        assert!(broke);
        ok(rbDestroyJoint(w, joint));
        assert!(!rbJoint_IsValid(w, joint));
        rbDestroyWorld(w);
    }
}

#[test]
fn vehicles_report_wheels_every_frame() {
    unsafe {
        let w = world(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        ground(w, false);
        let mut t = RbTransform::default();
        t.p.y = 1.5;
        let mut v = 0;
        ok(rbCreateVehiclePreset(w, c"sedan".as_ptr(), t, &mut v));
        let input = RbVehicleInput {
            forward: 1.0,
            ..Default::default()
        };
        ok(rbVehicle_SetInput(w, v, input));
        for _ in 0..60 {
            ok(rbWorld_Step(w));
        }
        let f = poll_all(w);
        assert_eq!(f.vehicle_count, 1);
        let out = &*f.vehicles;
        assert_eq!(out.vehicle, v);
        assert_eq!(out.wheel_count, 4);
        let wheels = std::slice::from_raw_parts(out.wheels, out.wheel_count as usize);
        assert!(wheels.iter().any(|wh| wh.omega.abs() > 0.1));
        let mut none = 0;
        assert_ne!(
            rbCreateVehiclePreset(w, c"no such car".as_ptr(), t, &mut none),
            RbResult::RB_OK
        );
        rbDestroyWorld(w);
    }
}
