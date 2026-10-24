//! The C ABI end to end, called the way a C host calls it.

#[test]
fn destructibles_break_and_split_through_the_abi() {
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.max_bodies = 16_384;
        def.max_colliders = 32_768;
        def.max_contacts = 262_144;
        def.max_joints = 32_768;
        def.max_candidates = 524_288;
        def.max_hull_vertices = 1 << 20;
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        ground(w, false);
        // Packs load too (their bonds carry stress-solver limits, unbreakable
        // under the bonds model); the arch GLB's bonds break.
        let pack = include_bytes!("../../../stress-oracle/packs/house-1story.json");
        let mut pack_asset = 0;
        ok(rbCreateDestructibleAssetFromPack(
            w,
            pack.as_ptr(),
            pack.len(),
            &mut pack_asset,
        ));
        let glb = include_bytes!("../../engine-webgpu/fracture/assets/masonry-arch.glb");
        let mut asset = 0;
        ok(rbCreateDestructibleAssetFromGlb(
            w,
            glb.as_ptr(),
            glb.len(),
            1,
            &mut asset,
        ));
        let mut ddef = rbDefaultDestructibleDef();
        ddef.asset = asset;
        let mut d = 0;
        ok(rbCreateDestructible(w, &ddef, &mut d));
        let mut chunks = 0;
        ok(rbDestructible_GetChunkCount(w, d, &mut chunks));
        assert!(chunks > 10);
        let mut body = 0;
        ok(rbDestructible_GetChunkBody(w, d, 0, &mut body));
        assert!(rbBody_IsValid(w, body));
        // A heavy ball through the middle of the structure.
        let desc = rubble::DestructibleAssetDesc::from_glb(glb, 1).unwrap();
        let n = desc.chunks.len() as f32;
        let c = desc.chunks.iter().fold([0.0f32; 3], |a, ch| {
            [
                a[0] + ch.pose.position[0] / n,
                a[1] + ch.pose.position[1] / n,
                a[2] + ch.pose.position[2] / n,
            ]
        });
        let mut bd = rbDefaultBodyDef();
        bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
        bd.transform.p = RbVec3 {
            x: c[0] - 20.0,
            y: c[1],
            z: c[2],
        };
        bd.linear_velocity = RbVec3 {
            x: 60.0,
            y: 0.0,
            z: 0.0,
        };
        let mut ball = 0;
        ok(rbCreateBody(w, &bd, &mut ball));
        let mut sd = rbDefaultShapeDef();
        sd.density = 2000.0 / (4.0 / 3.0 * std::f32::consts::PI * 0.216);
        let mut s = 0;
        ok(rbCreateSphereShape(
            w,
            ball,
            &sd,
            RbVec3::default(),
            0.6,
            &mut s,
        ));
        let (mut bonds, mut splits) = (0u32, 0u32);
        for _ in 0..120 {
            ok(rbWorld_Step(w));
            let f = poll_all(w);
            if !f.has_frame {
                continue;
            }
            for e in std::slice::from_raw_parts(f.destruction, f.destruction_count as usize) {
                match e.type_ {
                    RbDestructionType::RB_BONDS_BROKEN => {
                        assert_eq!(e.destructible, d);
                        bonds += e.bond_count;
                    }
                    RbDestructionType::RB_SPLIT => {
                        splits += 1;
                        let kids = std::slice::from_raw_parts(e.children, e.child_count as usize);
                        let n: u32 = kids.iter().map(|k| k.chunk_count).sum();
                        assert_eq!(n, e.parent.chunk_count, "children partition the parent");
                    }
                    _ => {}
                }
            }
        }
        assert!(bonds > 0 && splits > 0, "bonds {bonds}, splits {splits}");
        ok(rbDestroyDestructible(w, d));
        rbDestroyWorld(w);
    }
}

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

/// Before its first frame, a body reports the pose it was created with (not the identity), so a
/// host can place what it draws at load.
#[test]
fn a_new_body_reports_its_creation_pose() {
    unsafe {
        let w = world(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        let mut def = rbDefaultBodyDef();
        def.type_ = RbBodyType::RB_DYNAMIC_BODY;
        def.transform.p = RbVec3 { x: 3.0, y: 5.0, z: -2.0 };
        def.transform.q = RbQuat { x: 0.0, y: 0.38268343, z: 0.0, w: 0.9238795 };
        let mut b = 0;
        ok(rbCreateBody(w, &def, &mut b));
        let mut t = RbTransform::default();
        ok(rbBody_GetTransform(w, b, &mut t));
        assert!((t.p.x - 3.0).abs() < 1e-6 && (t.p.y - 5.0).abs() < 1e-6 && (t.p.z + 2.0).abs() < 1e-6, "{t:?}");
        assert!((t.q.y - 0.38268343).abs() < 1e-6, "{t:?}");
        let mut flat = [RbTransform::default()];
        ok(rbWorld_GetTransforms(w, &b, 1, flat.as_mut_ptr()));
        assert_eq!(flat[0], t);
        rbDestroyWorld(w);
    }
}

/// A body reporting only sleep and wake (RB_MOVES_SLEEP_WAKE) falls and sleeps exactly as one
/// reporting every move, but its frames carry only its first report and its sleep.
#[test]
fn move_reports_filter_frames_not_the_simulation() {
    unsafe {
        let w = world(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        ground(w, false);
        let mut bodies = [0u64; 2];
        for (k, x) in [-2.0f32, 2.0].into_iter().enumerate() {
            let mut def = rbDefaultBodyDef();
            def.type_ = RbBodyType::RB_DYNAMIC_BODY;
            def.transform.p = RbVec3 { x, y: 3.0, z: 0.0 };
            ok(rbCreateBody(w, &def, &mut bodies[k]));
            let mut s = 0;
            ok(rbCreateBoxShape(w, bodies[k], &rbDefaultShapeDef(), RbVec3 { x: 0.5, y: 0.5, z: 0.5 }, RbTransform::default(), &mut s));
        }
        let (all, quiet) = (bodies[0], bodies[1]);
        ok(rbBody_SetMoveReports(w, quiet, RbMoveReports::RB_MOVES_SLEEP_WAKE));
        let (mut all_moves, mut quiet_moves, mut quiet_slept) = (0, 0, false);
        for _ in 0..400 {
            ok(rbWorld_Step(w));
            let f = poll_all(w);
            if f.has_frame {
                for m in std::slice::from_raw_parts(f.moves, f.move_count as usize) {
                    if m.body == all {
                        all_moves += 1;
                    } else if m.body == quiet {
                        quiet_moves += 1;
                        quiet_slept |= m.fell_asleep;
                    }
                }
            }
        }
        assert!(all_moves > 20, "the default reports every move ({all_moves})");
        assert!(quiet_slept && quiet_moves <= 3, "sleep and wake only ({quiet_moves} moves)");
        let (mut a, mut q) = (RbTransform::default(), RbTransform::default());
        ok(rbBody_GetTransform(w, all, &mut a));
        ok(rbBody_GetTransform(w, quiet, &mut q));
        assert!((a.p.y - q.p.y).abs() < 1e-4, "both rest the same: {a:?} {q:?}");
        rbDestroyWorld(w);
    }
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
fn queued_rays_come_back_with_the_frame_in_host_units() {
    unsafe {
        let w = world(RbCoordinates::RB_COORDINATES_UNREAL);
        let g = ground(w, true);
        let down = |x: f32| RbRayInput {
            origin: RbVec3 {
                x,
                y: 0.0,
                z: 1000.0,
            },
            translation: RbVec3 {
                x: 0.0,
                y: 0.0,
                z: -2000.0,
            },
            ignore_body: 0,
        };
        let rays = [down(0.0), down(9000.0)];
        let mut batch = 0;
        ok(rbWorld_CastRays(w, rays.as_ptr(), 2, &mut batch));
        assert_ne!(batch, 0);
        ok(rbWorld_Step(w));
        let mut f: RbFrame = std::mem::zeroed();
        ok(rbWorld_Poll(w, true, true, &mut f));
        assert_eq!(f.ray_count, 1);
        let r = &*f.rays;
        assert_eq!(r.batch, batch);
        let hits = std::slice::from_raw_parts(r.hits, r.hit_count as usize);
        assert!(hits[0].hit && hits[0].body == g);
        assert!((hits[0].distance - 1000.0).abs() < 0.5, "{:?}", hits[0]);
        assert!((hits[0].point.z).abs() < 0.5 && hits[0].normal.z > 0.99);
        assert!(!hits[1].hit, "past the ground's edge");
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
        // Kinematic bodies take target transforms; other bodies refuse them.
        let mut kinematic = rbDefaultBodyDef();
        kinematic.type_ = RbBodyType::RB_KINEMATIC_BODY;
        ok(rbCreateBody(w, &kinematic, &mut b));
        ok(rbBody_SetTargetTransform(w, b, RbTransform::default()));
        let mut s = 0;
        ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut s));
        assert_eq!(
            rbBody_SetTargetTransform(w, s, RbTransform::default()),
            RbResult::RB_ERROR
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
