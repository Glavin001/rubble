//! The fracture lab's `threshold` mode through the C ABI, in Unreal coordinates: the
//! masonry arch and a cuboid house, shot as the lab shoots, break bonds and split into
//! new bodies that the host can query.

use rubble::DestructibleAssetDesc;
use rubble_ffi::coords::{Coords, RbCoordinates};
use rubble_ffi::*;
use std::ffi::CStr;
use std::ptr;

fn ok(r: RbResult) {
    if r != RbResult::RB_OK {
        let msg = unsafe { CStr::from_ptr(rbGetLastError()) };
        panic!("{r:?}: {}", msg.to_string_lossy());
    }
}

fn v3(v: [f32; 3]) -> RbVec3 {
    RbVec3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

/// The lab's shot at `target` (Rubble's frame) from 3 extents in front.
fn shot(lab: &engine_fracture::Scene, target: [f64; 3]) -> ([f32; 3], [f32; 3], f32, f32) {
    let origin = [target[0], target[1], target[2] + 3.0 * lab.extent.max(1.0)];
    let speed = engine_fracture::plan::default_shot_speed(lab.extent, lab.projectile_speed);
    let v = engine_fracture::plan::shot_velocity(
        target,
        lab.extent,
        lab.projectile_radius,
        origin,
        [0.0, 0.0, -1.0],
        Some(speed),
        lab.projectile_arc,
    );
    (
        origin.map(|x| x as f32),
        v.map(|x| x as f32),
        lab.projectile_radius as f32,
        lab.projectile_mass as f32,
    )
}

/// (bonds broken, splits, pieces that are live bodies).
unsafe fn run(
    make_asset: impl Fn(*mut RbWorld, *mut RbDestructibleAssetId),
    lab: &engine_fracture::Scene,
    target: [f64; 3],
) -> (u32, u32, u32) {
    let unreal = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
    let mut def = rbDefaultWorldDef();
    def.coordinates = RbCoordinates::RB_COORDINATES_UNREAL;
    def.gravity = v3([0.0, 0.0, -981.0]);
    def.contacts = RbContacts::RB_CONTACTS_CANONICAL;
    def.enable_sleep = false;
    def.max_bodies = 4096;
    def.max_colliders = 8192;
    def.max_contacts = 65_536;
    def.max_candidates = 262_144;
    def.max_hull_vertices = 1 << 20;
    let mut w = ptr::null_mut();
    ok(rbCreateWorld(&def, &mut w));
    let mut asset = 0;
    make_asset(w, &mut asset);
    let mut dt = 0.0;
    ok(rbWorld_UseDestructibleAssetSolverFor(
        w,
        asset,
        RbStrengthModel::RB_STRENGTH_THRESHOLD,
        &mut dt,
    ));
    assert!(
        (dt - 1.0 / 60.0).abs() < 1e-6,
        "the stress scene's 60 Hz: {dt}"
    );
    // The lab's ground, where the asset's stress scene puts it.
    let (mut has_ground, mut point) = (false, v3([0.0; 3]));
    ok(rbWorld_GetDestructibleAssetGround(
        w,
        asset,
        &mut has_ground,
        &mut point,
    ));
    assert!(has_ground);
    let mut g = 0;
    ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
    let mut plane = 0;
    ok(rbCreatePlaneShape(
        w,
        g,
        &rbDefaultShapeDef(),
        v3([0.0, 0.0, 1.0]),
        point.z,
        &mut plane,
    ));
    let mut dd = rbDefaultDestructibleDef();
    dd.asset = asset;
    dd.model = RbStrengthModel::RB_STRENGTH_THRESHOLD;
    dd.start_asleep = false;
    let mut id = 0;
    ok(rbCreateDestructible(w, &dd, &mut id));
    let mut chunks = 0;
    ok(rbDestructible_GetChunkCount(w, id, &mut chunks));
    assert!(chunks > 10);
    // The ball, in centimetres and Z up, its mass from its density.
    let (origin, v, r, m) = shot(lab, target);
    let mut bd = rbDefaultBodyDef();
    bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
    bd.transform.p = v3(unreal.position_out(origin));
    bd.linear_velocity = v3(unreal.linear_out(v));
    let mut ball = 0;
    ok(rbCreateBody(w, &bd, &mut ball));
    let r_cm = unreal.length_out(r);
    let mut sd = rbDefaultShapeDef();
    sd.density = m / (4.0 / 3.0 * std::f32::consts::PI * r_cm.powi(3));
    let mut s = 0;
    ok(rbCreateSphereShape(
        w,
        ball,
        &sd,
        v3([0.0; 3]),
        r_cm,
        &mut s,
    ));
    let p = rbDefaultProjectile();
    ok(rbBody_SetProjectile(w, ball, &p));
    let (mut broken, mut splits, mut pieces) = (0, 0, 0);
    for _ in 0..120 {
        ok(rbWorld_Step(w));
        let mut f: RbFrame = std::mem::zeroed();
        ok(rbWorld_Poll(w, true, true, &mut f));
        let events = std::slice::from_raw_parts(f.destruction, f.destruction_count as usize);
        for e in events {
            match e.type_ {
                RbDestructionType::RB_BONDS_BROKEN => {
                    assert_eq!(e.destructible, id);
                    broken += e.bond_count;
                }
                RbDestructionType::RB_SPLIT => {
                    splits += 1;
                    let children = std::slice::from_raw_parts(e.children, e.child_count as usize);
                    let total: u32 = children.iter().map(|c| c.chunk_count).sum();
                    assert_eq!(
                        total, e.parent.chunk_count,
                        "the children partition the parent"
                    );
                    for c in children {
                        // A body for every chunk (RbActor's contract), the piece's for each.
                        let bodies = std::slice::from_raw_parts(c.bodies, c.chunk_count as usize);
                        assert!(bodies.iter().all(|&b| b == bodies[0] && b != 0), "one live body per chunk: {bodies:?}");
                        let mut t = RbTransform::default();
                        ok(rbBody_GetTransform(w, bodies[0], &mut t));
                        assert!(t.p.x.is_finite() && t.p.z.is_finite());
                        pieces += 1;
                    }
                }
                _ => {}
            }
        }
    }
    // Every chunk is on a live body.
    for c in 0..chunks {
        let mut b = 0;
        ok(rbDestructible_GetChunkBody(w, id, c, &mut b));
        let mut t = RbTransform::default();
        ok(rbBody_GetTransform(w, b, &mut t));
    }
    ok(rbDestroyDestructible(w, id));
    rbDestroyWorld(w);
    (broken, splits, pieces)
}

#[test]
fn threshold_destructibles_break_and_split_through_the_abi() {
    unsafe {
        let glb = include_bytes!("../../engine-webgpu/fracture/assets/masonry-arch.glb");
        let desc = DestructibleAssetDesc::from_glb(glb, 1).unwrap();
        let keystone = desc
            .chunks
            .iter()
            .max_by(|a, b| a.pose.position[1].total_cmp(&b.pose.position[1]))
            .unwrap()
            .pose
            .position
            .map(|x| x as f64);
        let parsed = engine_fracture::parse_fracture_glb(glb).unwrap();
        let lab = engine_fracture::build_scene(&parsed, 1, None).unwrap();
        let arch = run(
            |w, out| {
                ok(rbCreateDestructibleAssetFromGlb(
                    w,
                    glb.as_ptr(),
                    glb.len(),
                    1,
                    out,
                ))
            },
            &lab,
            keystone,
        );
        assert!(arch.0 > 0 && arch.1 > 0 && arch.2 > 0, "arch: {arch:?}");

        let (lab, _) = engine_fracture::samples::cuboid_stress_scene(None).unwrap();
        let house = run(
            |w, out| ok(rbCreateDestructibleAssetCuboidHouse(w, out)),
            &lab,
            lab.center,
        );
        assert!(
            house.0 > 0 && house.1 > 0 && house.2 > 0,
            "house: {house:?}"
        );
        eprintln!("arch {arch:?}, house {house:?} (bonds broken, splits, pieces)");
    }
}
