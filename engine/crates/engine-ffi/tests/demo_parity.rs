//! The calls the demo pages need, through the C ABI: vehicle descriptors,
//! damage and telemetry, soft bodies from presets, live joint drives, world
//! and shape edits, cuboid houses and debug drawing.

use rubble_ffi::coords::RbCoordinates;
use rubble_ffi::*;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

fn ok(r: RbResult) {
    if r != RbResult::RB_OK {
        let msg = unsafe { CStr::from_ptr(rbGetLastError()) };
        panic!("{r:?}: {}", msg.to_string_lossy());
    }
}

unsafe fn world(soft: bool) -> *mut RbWorld {
    let mut def = rbDefaultWorldDef();
    if soft {
        def.enable_soft_bodies = true;
        def.contacts = RbContacts::RB_CONTACTS_CANONICAL;
        def.max_bodies = 8192;
        def.max_colliders = 8192;
        def.max_contacts = 131_072;
        def.max_candidates = 262_144;
        def.max_joints = 4096;
    }
    let mut w = ptr::null_mut();
    ok(rbCreateWorld(&def, &mut w));
    w
}

fn v3(x: f32, y: f32, z: f32) -> RbVec3 {
    RbVec3 { x, y, z }
}

fn at(x: f32, y: f32, z: f32) -> RbTransform {
    RbTransform {
        p: v3(x, y, z),
        ..RbTransform::default()
    }
}

unsafe fn ground(w: *mut RbWorld) -> RbShapeId {
    let mut g = 0;
    ok(rbCreateBody(
        w,
        &RbBodyDef {
            transform: at(0.0, -0.5, 0.0),
            ..rbDefaultBodyDef()
        },
        &mut g,
    ));
    let mut s = 0;
    ok(rbCreateBoxShape(
        w,
        g,
        &rbDefaultShapeDef(),
        v3(60.0, 0.5, 60.0),
        RbTransform::default(),
        &mut s,
    ));
    s
}

unsafe fn json(f: impl Fn(*mut c_char, usize, *mut usize) -> RbResult) -> String {
    let mut n = 0;
    ok(f(ptr::null_mut(), 0, &mut n));
    let mut buf = vec![0u8; n + 1];
    ok(f(buf.as_mut_ptr().cast(), buf.len(), &mut n));
    CStr::from_bytes_with_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned()
}

unsafe fn steps(w: *mut RbWorld, n: usize) -> RbFrame {
    for _ in 0..n {
        ok(rbWorld_Step(w));
    }
    let mut f = std::mem::zeroed();
    ok(rbWorld_Poll(w, true, true, &mut f));
    f
}

#[test]
fn vehicles_from_json_take_damage_and_report_it() {
    unsafe {
        let w = world(false);
        ground(w);
        let text = json(|b, c, n| rbGetVehiclePresetJson(c"destructible_sedan".as_ptr(), b, c, n));
        assert!(
            text.contains("wheels"),
            "a VehicleDesc: {}",
            &text[..80.min(text.len())]
        );
        let desc = CString::new(text).unwrap();
        let mut v = 0;
        ok(rbCreateVehicleFromJson(
            w,
            desc.as_ptr(),
            at(0.0, 1.5, 0.0),
            &mut v,
        ));
        let input = RbVehicleInput {
            forward: 1.0,
            manual_gear: true,
            gear: 1,
            ..Default::default()
        };
        ok(rbVehicle_SetInput(w, v, input));
        let f = steps(w, 60);
        let out = &*f.vehicles;
        assert_eq!(out.gear, 1, "manual first gear");
        assert!(out.mass > 0.0 && out.engine_connected);
        let wheels = std::slice::from_raw_parts(out.wheels, out.wheel_count as usize);
        assert!(wheels.iter().all(|wh| wh.radius > 0.0 && !wh.detached));
        assert!(wheels.iter().any(|wh| wh.suspension_force > 0.0));

        let mut count = 0;
        ok(rbVehicle_GetMountLoads(
            w,
            v,
            ptr::null_mut(),
            0,
            &mut count,
        ));
        assert!(count > 0, "the destructible sedan has breakable mounts");
        let mut loads = vec![RbMountLoad::default(); count as usize];
        ok(rbVehicle_GetMountLoads(
            w,
            v,
            loads.as_mut_ptr(),
            count,
            &mut count,
        ));
        assert!(loads.iter().all(|l| l.break_force > 0.0));

        let mut cut = rbDefaultFunctionalState();
        cut.engine_connected = false;
        ok(rbVehicle_SetFunctionalState(w, v, cut));
        ok(rbVehicle_Damage(
            w,
            v,
            RbVehicleDamageType::RB_DETACH_WHEEL,
            0,
        ));
        let f = steps(w, 30);
        let out = &*f.vehicles;
        let wheels = std::slice::from_raw_parts(out.wheels, out.wheel_count as usize);
        assert!(wheels[0].detached, "wheel 0 came off");
        assert!(!out.engine_connected);
        assert_eq!(
            rbVehicle_Damage(w, 99, RbVehicleDamageType::RB_DETACH_WHEEL, 0),
            RbResult::RB_ERROR
        );
        rbDestroyWorld(w);
    }
}

#[test]
fn joints_drive_live_and_the_world_edits_in_place() {
    unsafe {
        let w = world(false);
        let floor = ground(w);
        ok(rbShape_SetFriction(w, floor, 0.3, 0.0));
        ok(rbShape_SetFilter(w, floor, 1, 0xFFFF, 0));
        ok(rbWorld_SetGravity(w, v3(0.0, 0.0, 0.0)));
        let mut anchor = 0;
        ok(rbCreateBody(
            w,
            &RbBodyDef {
                transform: at(0.0, 5.0, 0.0),
                ..rbDefaultBodyDef()
            },
            &mut anchor,
        ));
        let mut wheel = 0;
        ok(rbCreateBody(
            w,
            &RbBodyDef {
                type_: RbBodyType::RB_DYNAMIC_BODY,
                transform: at(0.0, 5.0, 0.0),
                ..rbDefaultBodyDef()
            },
            &mut wheel,
        ));
        let mut s = 0;
        ok(rbCreateSphereShape(
            w,
            wheel,
            &rbDefaultShapeDef(),
            RbVec3::default(),
            0.5,
            &mut s,
        ));
        let mut d = rbDefaultRevoluteJointDef();
        d.base.body_a = anchor;
        d.base.body_b = wheel;
        d.enable_motor = true;
        d.motor_speed = 2.0;
        d.max_motor_torque = 1.0e4;
        let mut j = 0;
        ok(rbCreateRevoluteJoint(w, &d, &mut j));
        let spin = |w| {
            let (mut lin, mut ang) = (RbVec3::default(), RbVec3::default());
            ok(rbBody_GetVelocity(w, wheel, &mut lin, &mut ang));
            (ang.x * ang.x + ang.y * ang.y + ang.z * ang.z).sqrt()
        };
        steps(w, 60);
        assert!((spin(w) - 2.0).abs() < 0.2, "motor at 2 rad/s: {}", spin(w));
        ok(rbJoint_SetDrive(w, j, 6.0, 0.0));
        steps(w, 60);
        assert!((spin(w) - 6.0).abs() < 0.5, "motor at 6 rad/s: {}", spin(w));
        let mut awake = false;
        ok(rbBody_IsAwake(w, wheel, &mut awake));
        assert!(awake);

        // A plastic weld is accepted.
        let mut weld = rbDefaultWeldJointDef();
        weld.base.body_a = anchor;
        weld.base.body_b = wheel;
        weld.yield_torque = 50.0;
        weld.break_bend = 1.0;
        let mut wj = 0;
        ok(rbCreateWeldJoint(w, &weld, &mut wj));
        rbDestroyWorld(w);
    }
}

#[test]
fn soft_presets_spawn_with_settings_obstacles_and_damage() {
    unsafe {
        let w = world(true);
        ground(w);
        let settings = json(|b, c, n| rbGetCalibratedSoftSettingsJson(60.0, b, c, n));
        assert!(settings.contains("rigid_mode"));
        let settings = CString::new(settings).unwrap();
        ok(rbWorld_SetSoftSettingsJson(w, settings.as_ptr()));
        // A wall the car drives into, as a soft obstacle.
        let mut wall = 0;
        ok(rbCreateBody(
            w,
            &RbBodyDef {
                transform: at(8.0, 1.0, 0.0),
                ..rbDefaultBodyDef()
            },
            &mut wall,
        ));
        let mut ws = 0;
        ok(rbCreateBoxShape(
            w,
            wall,
            &rbDefaultShapeDef(),
            v3(0.3, 1.0, 3.0),
            RbTransform::default(),
            &mut ws,
        ));
        ok(rbSoft_AddObstacle(w, ws));
        for (k, name) in [c"procedural_car", c"dune_buggy"].into_iter().enumerate() {
            let desc = json(|b, c, n| rbGetSoftVehiclePresetJson(name.as_ptr(), 0.0, b, c, n));
            let desc = CString::new(desc).unwrap();
            let mut s = 0;
            ok(rbCreateSoftVehicle(
                w,
                desc.as_ptr(),
                ptr::null(),
                at(0.0, 1.0, if k == 0 { -10.0 } else { 10.0 }),
                v3(0.0, 0.0, 0.0),
                &mut s,
            ));
            steps(w, 20);
            let mut dmg = RbSoftDamage::default();
            ok(rbSoft_GetDamage(w, s, &mut dmg));
            assert!(dmg.beams > 0, "{name:?} has beams");
        }
        rbDestroyWorld(w);
    }
}

#[derive(Default)]
struct Counts {
    boxes: u32,
    spheres: u32,
}

unsafe extern "C" fn on_box(ctx: *mut c_void, _: RbTransform, half: RbVec3, _: RbColor) {
    assert!(half.x > 0.0 && half.y > 0.0 && half.z > 0.0);
    (*ctx.cast::<Counts>()).boxes += 1;
}

unsafe extern "C" fn on_sphere(ctx: *mut c_void, _: RbVec3, _: f32, _: RbColor) {
    (*ctx.cast::<Counts>()).spheres += 1;
}

#[test]
fn cuboid_houses_load_and_debug_draw_calls_back() {
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.max_bodies = 8192;
        def.max_colliders = 8192;
        def.max_joints = 8192;
        def.max_contacts = 131_072;
        def.max_candidates = 262_144;
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        ground(w);
        let mut asset = 0;
        ok(rbCreateDestructibleAssetCuboidHouse(w, &mut asset));
        let mut dd = rbDefaultDestructibleDef();
        dd.asset = asset;
        let mut house = 0;
        ok(rbCreateDestructible(w, &dd, &mut house));
        let mut chunks = 0;
        ok(rbDestructible_GetChunkCount(w, house, &mut chunks));
        assert!(chunks > 50);
        steps(w, 2);
        let mut counts = Counts::default();
        let draw = RbDebugDraw {
            context: (&mut counts as *mut Counts).cast(),
            draw_box: Some(on_box),
            draw_sphere: Some(on_sphere),
            draw_capsule: None,
            draw_mesh: None,
            draw_transform: None,
            draw_shapes: true,
            draw_body_transforms: false,
        };
        ok(rbWorld_DebugDraw(w, &draw));
        assert!(
            counts.boxes > chunks,
            "ground plus every chunk: {}",
            counts.boxes
        );
        assert_eq!(counts.spheres, 0);
        rbDestroyWorld(w);
    }
}
