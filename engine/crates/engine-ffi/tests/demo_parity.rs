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

#[test]
fn the_fracture_lab_setup_works_in_unreal_units() {
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.coordinates = RbCoordinates::RB_COORDINATES_UNREAL;
        def.gravity = v3(0.0, 0.0, -981.0);
        def.max_bodies = 8192;
        def.max_colliders = 8192;
        def.max_joints = 8192;
        def.max_contacts = 131_072;
        def.max_candidates = 262_144;
        def.max_hull_vertices = 1 << 20;
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let mut g = 0;
        ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
        let mut plane = 0;
        ok(rbCreatePlaneShape(
            w,
            g,
            &rbDefaultShapeDef(),
            v3(0.0, 0.0, 1.0),
            0.0,
            &mut plane,
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
        let mut dt = 0.0;
        ok(rbWorld_UseDestructibleAssetSolver(w, asset, &mut dt));
        assert!((dt - 1.0 / 120.0).abs() < 1e-6, "the lab's 120 Hz: {dt}");
        let mut settled = true;
        ok(rbWorld_PreloadDestructibleAsset(w, asset, &mut settled));
        let mut dd = rbDefaultDestructibleDef();
        dd.asset = asset;
        let mut arch = 0;
        ok(rbCreateDestructible(w, &dd, &mut arch));
        let f = steps(w, 120);
        let events = std::slice::from_raw_parts(f.destruction, f.destruction_count as usize);
        assert!(
            events
                .iter()
                .all(|e| e.type_ != RbDestructionType::RB_BONDS_BROKEN),
            "it holds its own weight"
        );
        // A chunk stands above the plane (z up, centimetres).
        let mut body = 0;
        ok(rbDestructible_GetChunkBody(w, arch, 0, &mut body));
        let mut t = RbTransform::default();
        ok(rbBody_GetTransform(w, body, &mut t));
        assert!(t.p.z.is_finite() && t.p.z > -5.0, "{t:?}");
        rbDestroyWorld(w);
    }
}

#[test]
fn contact_and_hit_events_arrive_in_unreal_units() {
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.coordinates = RbCoordinates::RB_COORDINATES_UNREAL;
        def.gravity = v3(0.0, 0.0, -981.0);
        def.hit_event_threshold = 100.0; // 1 m/s
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let mut g = 0;
        ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
        let mut plane = 0;
        ok(rbCreatePlaneShape(
            w,
            g,
            &rbDefaultShapeDef(),
            v3(0.0, 0.0, 1.0),
            0.0,
            &mut plane,
        ));
        let mut bd = rbDefaultBodyDef();
        bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
        bd.transform.p = v3(0.0, 0.0, 250.0);
        let mut b = 0;
        ok(rbCreateBody(w, &bd, &mut b));
        let mut sd = rbDefaultShapeDef();
        sd.density = 0.0005;
        sd.enable_contact_events = true;
        sd.enable_hit_events = true;
        let mut s = 0;
        ok(rbCreateBoxShape(
            w,
            b,
            &sd,
            v3(50.0, 50.0, 50.0),
            RbTransform::default(),
            &mut s,
        ));
        let (mut begins, mut hits) = (Vec::new(), Vec::new());
        for _ in 0..120 {
            ok(rbWorld_Step(w));
            let mut f: RbFrame = std::mem::zeroed();
            ok(rbWorld_Poll(w, true, false, &mut f));
            if f.contact_begin_count > 0 {
                begins.extend_from_slice(std::slice::from_raw_parts(
                    f.contact_begin,
                    f.contact_begin_count as usize,
                ));
            }
            if f.hit_count > 0 {
                hits.extend_from_slice(std::slice::from_raw_parts(f.hits, f.hit_count as usize));
            }
        }
        assert_eq!(begins.len(), 1, "{begins:?}");
        let e = begins[0];
        assert!(e.shape_a == s || e.shape_b == s);
        let up = if e.shape_a == plane { 1.0 } else { -1.0 };
        assert!(e.normal.z * up > 0.9, "z up: {e:?}");
        assert_eq!(hits.len(), 1);
        // Fell 2 m (through nothing): about 630 cm/s.
        assert!(
            (hits[0].approach_speed - 630.0).abs() < 80.0,
            "{:?}",
            hits[0]
        );
        rbDestroyWorld(w);
    }
}

#[test]
fn sensors_report_through_the_abi() {
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.coordinates = RbCoordinates::RB_COORDINATES_UNREAL;
        def.gravity = v3(0.0, 0.0, -981.0);
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let mut g = 0;
        ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
        let mut floor = 0;
        ok(rbCreatePlaneShape(
            w,
            g,
            &rbDefaultShapeDef(),
            v3(0.0, 0.0, 1.0),
            0.0,
            &mut floor,
        ));
        // A trigger volume between 300 and 500 cm.
        let mut zone = 0;
        let mut zd = rbDefaultBodyDef();
        zd.transform.p = v3(0.0, 0.0, 400.0);
        ok(rbCreateBody(w, &zd, &mut zone));
        let mut sd = rbDefaultShapeDef();
        sd.is_sensor = true;
        sd.enable_sensor_events = true;
        let mut trigger = 0;
        ok(rbCreateBoxShape(
            w,
            zone,
            &sd,
            v3(200.0, 200.0, 100.0),
            RbTransform::default(),
            &mut trigger,
        ));
        let mut bd = rbDefaultBodyDef();
        bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
        bd.transform.p = v3(0.0, 0.0, 700.0);
        let mut b = 0;
        ok(rbCreateBody(w, &bd, &mut b));
        let mut box_shape = 0;
        let mut bs = rbDefaultShapeDef();
        bs.density = 0.0005;
        ok(rbCreateBoxShape(
            w,
            b,
            &bs,
            v3(50.0, 50.0, 50.0),
            RbTransform::default(),
            &mut box_shape,
        ));
        let (mut begins, mut ends) = (Vec::new(), Vec::new());
        for _ in 0..180 {
            ok(rbWorld_Step(w));
            let mut f: RbFrame = std::mem::zeroed();
            ok(rbWorld_Poll(w, true, false, &mut f));
            if f.sensor_begin_count > 0 {
                begins.extend_from_slice(std::slice::from_raw_parts(
                    f.sensor_begin,
                    f.sensor_begin_count as usize,
                ));
            }
            if f.sensor_end_count > 0 {
                ends.extend_from_slice(std::slice::from_raw_parts(
                    f.sensor_end,
                    f.sensor_end_count as usize,
                ));
            }
        }
        assert_eq!(begins.len(), 1, "{begins:?}");
        assert_eq!(ends.len(), 1, "{ends:?}");
        assert_eq!(begins[0].sensor_shape, trigger);
        assert_eq!(begins[0].visitor_shape, box_shape);
        let mut t = RbTransform::default();
        ok(rbBody_GetTransform(w, b, &mut t));
        assert!(
            (t.p.z - 50.0).abs() < 6.0,
            "fell through onto the floor: {t:?}"
        );
        rbDestroyWorld(w);
    }
}

#[test]
fn damping_gravity_scale_and_locks_through_the_abi() {
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.coordinates = RbCoordinates::RB_COORDINATES_UNREAL;
        def.gravity = v3(0.0, 0.0, -981.0);
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let make = |w: *mut RbWorld, x: f32, edit: &dyn Fn(&mut RbBodyDef)| {
            let mut bd = rbDefaultBodyDef();
            bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
            bd.transform.p = v3(x, 0.0, 1000.0);
            edit(&mut bd);
            let mut b = 0;
            ok(rbCreateBody(w, &bd, &mut b));
            let mut sd = rbDefaultShapeDef();
            sd.density = 0.0005;
            let mut s = 0;
            ok(rbCreateBoxShape(
                w,
                b,
                &sd,
                v3(50.0, 50.0, 50.0),
                RbTransform::default(),
                &mut s,
            ));
            b
        };
        let normal = make(w, 0.0, &|_| {});
        let locked = make(w, 300.0, &|d| d.motion_locks = RB_LOCK_LINEAR_Z);
        let light = make(w, 600.0, &|d| d.gravity_scale = 0.5);
        let damped = make(w, 900.0, &|_| {});
        ok(rbBody_SetDamping(w, damped, 5.0, 0.0));
        for _ in 0..30 {
            ok(rbWorld_Step(w));
        }
        let mut f: RbFrame = std::mem::zeroed();
        ok(rbWorld_Poll(w, true, true, &mut f));
        let z = |b| {
            let mut t = RbTransform::default();
            ok(rbBody_GetTransform(w, b, &mut t));
            1000.0 - t.p.z
        };
        let drop = z(normal);
        assert!(drop > 100.0, "fell {drop} cm");
        assert!(
            z(locked).abs() < 0.01,
            "locked along Unreal Z: {}",
            z(locked)
        );
        assert!(
            (z(light) / drop - 0.5).abs() < 0.02,
            "half gravity: {}",
            z(light)
        );
        assert!(z(damped) < 0.9 * drop, "damped: {}", z(damped));
        rbDestroyWorld(w);
    }
}

#[test]
fn mass_data_and_getters_through_the_abi() {
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.coordinates = RbCoordinates::RB_COORDINATES_UNREAL;
        def.gravity = v3(0.0, 0.0, -981.0);
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let mut bd = rbDefaultBodyDef();
        bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
        bd.user_data = 42;
        let mut b = 0;
        ok(rbCreateBody(w, &bd, &mut b));
        // A 100 x 200 x 400 cm box of 0.0005 kg/cm³: 4000 kg.
        let mut sd = rbDefaultShapeDef();
        sd.density = 0.0005;
        let mut s = 0;
        ok(rbCreateBoxShape(
            w,
            b,
            &sd,
            v3(50.0, 100.0, 200.0),
            RbTransform::default(),
            &mut s,
        ));
        let mut m = RbMassData::default();
        ok(rbBody_GetMassData(w, b, &mut m));
        assert!((m.mass - 4000.0).abs() < 1.0, "{m:?}");
        // I_xx = m (b² + c²) / 12 with b, c the y and z sizes (cm).
        let ixx = 4000.0 * (200.0f32.powi(2) + 400.0f32.powi(2)) / 12.0;
        assert!(
            (m.rotational_inertia[0].x - ixx).abs() < 1e-3 * ixx,
            "{m:?}"
        );
        let izz = 4000.0 * (100.0f32.powi(2) + 200.0f32.powi(2)) / 12.0;
        assert!(
            (m.rotational_inertia[2].z - izz).abs() < 1e-3 * izz,
            "{m:?}"
        );
        let mut doubled = m;
        doubled.mass *= 2.0;
        for r in &mut doubled.rotational_inertia {
            r.x *= 2.0;
            r.y *= 2.0;
            r.z *= 2.0;
        }
        ok(rbBody_SetMassData(w, b, doubled));
        let mut again = RbMassData::default();
        ok(rbBody_GetMassData(w, b, &mut again));
        assert!((again.mass - 8000.0).abs() < 1.0);
        assert!((again.rotational_inertia[0].x - 2.0 * ixx).abs() < 1e-3 * ixx);
        let mut t = RbBodyType::RB_STATIC_BODY;
        ok(rbBody_GetType(w, b, &mut t));
        assert_eq!(t, RbBodyType::RB_DYNAMIC_BODY);
        let mut u = 0;
        ok(rbBody_GetUserData(w, b, &mut u));
        assert_eq!(u, 42);
        ok(rbWorld_Step(w));
        let mut ms = -1.0;
        ok(rbWorld_GetProfileMs(w, &mut ms));
        assert!(ms >= 0.0);
        rbDestroyWorld(w);
    }
}

#[test]
fn soft_beams_for_the_beam_view() {
    unsafe {
        let w = world(true);
        ground(w);
        let desc =
            json(|b, c, n| rbGetSoftVehiclePresetJson(c"procedural_car".as_ptr(), 0.0, b, c, n));
        let desc = CString::new(desc).unwrap();
        let mut s = 0;
        ok(rbCreateSoftVehicle(
            w,
            desc.as_ptr(),
            ptr::null(),
            at(0.0, 1.0, 0.0),
            v3(0.0, 0.0, 0.0),
            &mut s,
        ));
        ok(rbSoft_EnableOutput(w, s, true));
        let f = steps(w, 10);
        let nodes = (*f.soft).node_count;
        let mut count = 0;
        ok(rbSoft_GetBeams(w, s, ptr::null_mut(), 0, &mut count));
        assert!(count > 50, "{count} beams");
        let mut beams = vec![RbSoftBeam::default(); count as usize];
        ok(rbSoft_GetBeams(w, s, beams.as_mut_ptr(), count, &mut count));
        assert!(beams.iter().all(|b| b.node_a < nodes && b.node_b < nodes));
        assert!(beams.iter().all(|b| b.spawn_length > 0.0 && !b.broken));
        assert!(beams.iter().any(|b| b.visible));
        rbDestroyWorld(w);
    }
}

#[test]
fn grab_and_drag_with_a_kinematic_mouse_body() {
    unsafe {
        let w = world(false);
        ground(w);
        let mut bd = rbDefaultBodyDef();
        bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
        bd.transform.p = v3(0.0, 0.5, 0.0);
        let mut b = 0;
        ok(rbCreateBody(w, &bd, &mut b));
        let mut s = 0;
        ok(rbCreateBoxShape(
            w,
            b,
            &rbDefaultShapeDef(),
            v3(0.5, 0.5, 0.5),
            RbTransform::default(),
            &mut s,
        ));
        // The mouse: a kinematic body at the grab point, jointed to the box.
        let mut md = rbDefaultBodyDef();
        md.type_ = RbBodyType::RB_KINEMATIC_BODY;
        md.transform.p = v3(0.0, 0.5, 0.0);
        let mut mouse = 0;
        ok(rbCreateBody(w, &md, &mut mouse));
        let mut jd = rbDefaultSphericalJointDef();
        jd.base.body_a = mouse;
        jd.base.body_b = b;
        let mut j = 0;
        ok(rbCreateSphericalJoint(w, &jd, &mut j));
        // Drag up 2 m over a second.
        for k in 1..=60 {
            let t = at(0.0, 0.5 + 2.0 * k as f32 / 60.0, 0.0);
            ok(rbBody_SetTargetTransform(w, mouse, t));
            ok(rbWorld_Step(w));
        }
        steps(w, 1);
        let mut t = RbTransform::default();
        ok(rbBody_GetTransform(w, b, &mut t));
        assert!((t.p.y - 2.5).abs() < 0.1, "the box came along: {t:?}");
        // Let go.
        ok(rbDestroyJoint(w, j));
        steps(w, 120);
        ok(rbBody_GetTransform(w, b, &mut t));
        assert!((t.p.y - 0.5).abs() < 0.06, "and dropped: {t:?}");
        rbDestroyWorld(w);
    }
}

#[test]
fn overlap_queries_and_explosions_in_unreal_units() {
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.coordinates = RbCoordinates::RB_COORDINATES_UNREAL;
        def.gravity = v3(0.0, 0.0, -981.0);
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let mut g = 0;
        ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
        let mut floor = 0;
        ok(rbCreatePlaneShape(
            w,
            g,
            &rbDefaultShapeDef(),
            v3(0.0, 0.0, 1.0),
            0.0,
            &mut floor,
        ));
        let make = |x: f32, y: f32| {
            let mut bd = rbDefaultBodyDef();
            bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
            bd.transform.p = v3(x, y, 50.0);
            let mut b = 0;
            ok(rbCreateBody(w, &bd, &mut b));
            let mut sd = rbDefaultShapeDef();
            sd.density = 0.0005;
            let mut s = 0;
            ok(rbCreateBoxShape(
                w,
                b,
                &sd,
                v3(50.0, 50.0, 50.0),
                RbTransform::default(),
                &mut s,
            ));
            (b, s)
        };
        let (east, east_shape) = make(300.0, 0.0);
        let (_far, far_shape) = make(3000.0, 0.0);
        steps(w, 2);
        // A box around the east crate only (above the floor).
        let mut q = 0;
        ok(rbWorld_OverlapAABB(
            w,
            v3(150.0, -150.0, 10.0),
            v3(450.0, 150.0, 200.0),
            0xFFFF,
            0xFFFF,
            &mut q,
        ));
        let f = steps(w, 1);
        assert_eq!(f.overlap_count, 1);
        let r = &*f.overlaps;
        assert_eq!(r.query, q);
        let shapes = std::slice::from_raw_parts(r.shapes, r.shape_count as usize);
        assert_eq!(shapes, &[east_shape], "{shapes:?}");
        assert!(!shapes.contains(&far_shape));
        // An explosion at the origin pushes the east crate along +X.
        let mut e = rbDefaultExplosionDef(RbCoordinates::RB_COORDINATES_UNREAL);
        e.position = v3(0.0, 0.0, 50.0);
        e.radius = 400.0;
        e.falloff = 200.0;
        e.impulse_per_area = 0.2; // kg cm/s per cm²
        ok(rbWorld_Explode(w, &e));
        steps(w, 1);
        let (mut lin, mut ang) = (RbVec3::default(), RbVec3::default());
        ok(rbBody_GetVelocity(w, east, &mut lin, &mut ang));
        // 500 kg, bounding radius 86.6 cm: pi r² = 23562 cm² at 0.2 gives
        // 4712 kg cm/s, so 9.42 cm/s.
        let expected = 0.2 * std::f32::consts::PI * (3.0f32 * 50.0 * 50.0) / 500.0;
        assert!(
            (lin.x - expected).abs() < 0.03 * expected,
            "pushed along +X: {lin:?} vs {expected}"
        );
        assert!(lin.y.abs() < 1e-3, "{lin:?}");
        rbDestroyWorld(w);
    }
}

#[test]
fn characters_casts_and_projectiles_in_unreal_units() {
    unsafe {
        let unreal = RbCoordinates::RB_COORDINATES_UNREAL;
        let mut def = rbDefaultWorldDef();
        def.coordinates = unreal;
        def.gravity = v3(0.0, 0.0, -981.0);
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let static_box = |p: RbVec3, half: RbVec3| {
            let mut bd = rbDefaultBodyDef();
            bd.transform.p = p;
            let mut b = 0;
            ok(rbCreateBody(w, &bd, &mut b));
            let mut s = 0;
            ok(rbCreateBoxShape(
                w,
                b,
                &rbDefaultShapeDef(),
                half,
                RbTransform::default(),
                &mut s,
            ));
            (b, s)
        };
        let (ground_body, ground_shape) = static_box(v3(0.0, 0.0, -50.0), v3(4000.0, 4000.0, 50.0));
        let (_, wall_shape) = static_box(v3(1000.0, 0.0, 200.0), v3(5.0, 200.0, 200.0));

        // A default character (30 cm radius, 60 cm half height) lands and walks.
        let mut cd = rbDefaultCharacterDef(unreal);
        assert!((cd.radius - 30.0).abs() < 1e-4 && (cd.jump_speed - 500.0).abs() < 1e-3);
        cd.position = v3(0.0, 0.0, 150.0);
        cd.user_data = 11;
        let mut c = 0;
        ok(rbWorld_CreateCharacter(w, &cd, &mut c));
        assert!(rbCharacter_IsValid(w, c));
        let mut landed = 0;
        for _ in 0..60 {
            let f = steps(w, 1);
            let events =
                std::slice::from_raw_parts(f.character_events, f.character_event_count as usize);
            landed += events
                .iter()
                .filter(|e| {
                    e.type_ == RbCharacterEventType::RB_CHARACTER_LANDED && e.character == c
                })
                .count();
        }
        assert_eq!(landed, 1);
        let mut s = std::mem::zeroed::<RbCharacterOutput>();
        let mut known = false;
        ok(rbCharacter_GetState(w, c, &mut s, &mut known));
        assert!(known && s.ground == RbGroundState::RB_ON_GROUND && s.user_data == 11);
        assert_eq!(s.ground_body, ground_body);
        assert!(
            (s.position.z - 90.0).abs() < 3.0,
            "stands on the floor: {:?}",
            s.position
        );
        ok(rbCharacter_SetInput(w, c, v3(200.0, 0.0, 0.0), false));
        steps(w, 60);
        ok(rbCharacter_GetState(w, c, &mut s, &mut known));
        assert!(
            (s.position.x - 200.0).abs() < 25.0 && s.position.y.abs() < 1.0,
            "walked forward: {:?}",
            s.position
        );
        assert!((s.velocity.x - 200.0).abs() < 5.0, "{:?}", s.velocity);

        // A sphere cast down from above the character, ignoring its capsule,
        // reaches the floor.
        let mut body = 0;
        ok(rbCharacter_GetBody(w, c, &mut body));
        let mut cast = rbDefaultCastDef(unreal);
        cast.origin = v3(s.position.x, s.position.y, 500.0);
        cast.direction = v3(0.0, 0.0, -1.0);
        cast.max_distance = 1000.0;
        cast.shape = RbCastShape::RB_CAST_SPHERE;
        cast.radius = 10.0;
        cast.ignore_body = body;
        let mut id = 0;
        ok(rbWorld_Cast(w, &cast, &mut id));
        let f = steps(w, 1);
        assert_eq!(f.cast_count, 1);
        let r = &*f.casts;
        assert!(
            r.cast == id && r.hit.hit && r.hit.shape == ground_shape,
            "{r:?}"
        );
        assert!((r.hit.distance - 490.0).abs() < 1.0, "{r:?}");
        assert!((r.hit.normal.z - 1.0).abs() < 1e-3, "{r:?}");

        // 10 g at 400 m/s (800 J) through a 10 cm wall costing 2000 J/m.
        let mut pd = rbDefaultProjectileDef(unreal);
        pd.position = v3(800.0, 0.0, 200.0);
        pd.velocity = v3(40_000.0, 0.0, 0.0);
        pd.mass = 0.01;
        pd.penetration_cost = 2000.0 * 100.0; // J/m in kg cm/s²
        pd.user_data = 5;
        let mut p = 0;
        ok(rbWorld_SpawnProjectile(w, &pd, &mut p));
        let f = steps(w, 1);
        let events =
            std::slice::from_raw_parts(f.projectile_events, f.projectile_event_count as usize);
        assert_eq!(events.len(), 1, "{events:?}");
        let e = events[0];
        assert!(
            e.projectile == p && e.shape == wall_shape && !e.stopped && e.user_data == 5,
            "{e:?}"
        );
        assert!((e.thickness - 10.0).abs() < 0.1, "{e:?}");
        let out = &*f.projectiles;
        // 800 J less 200 J, in kg cm²/s².
        assert!(
            out.alive && (out.energy - 600.0 * 1e4).abs() < 0.01 * 600.0 * 1e4,
            "{out:?}"
        );

        ok(rbCharacter_Destroy(w, c));
        assert!(!rbCharacter_IsValid(w, c));
        assert_eq!(
            rbCharacter_SetInput(w, c, v3(0.0, 0.0, 0.0), true),
            RbResult::RB_STALE_HANDLE
        );
        rbDestroyWorld(w);
    }
}
