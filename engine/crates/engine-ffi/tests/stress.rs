//! The fracture lab's `stress` mode (blast_parity) through the C ABI, in Unreal
//! coordinates: the masonry arch and a cuboid house, shot as the lab shoots (an impactor
//! launched at the destructible), break bonds and split into new bodies that the host
//! can query.

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
    strength_scale: f32,
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
        RbStrengthModel::RB_STRENGTH_STRESS,
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
    dd.model = RbStrengthModel::RB_STRENGTH_STRESS;
    dd.start_asleep = false;
    dd.strength_scale = strength_scale;
    let mut id = 0;
    ok(rbCreateDestructible(w, &dd, &mut id));
    let mut chunks = 0;
    ok(rbDestructible_GetChunkCount(w, id, &mut chunks));
    assert!(chunks > 10);
    // The shot, in centimetres and Z up.
    let (origin, v, r, m) = shot(lab, target);
    let mut sd = rbDefaultImpactorDef(RbCoordinates::RB_COORDINATES_UNREAL);
    sd.radius = unreal.length_out(r);
    sd.mass = m;
    sd.transform.p = v3(unreal.position_out(origin));
    sd.linear_velocity = v3(unreal.linear_out(v));
    let mut shot_index = u32::MAX;
    ok(rbDestructible_LaunchImpactor(w, id, &sd, &mut shot_index));
    let mut bad = sd;
    bad.mass = 0.0;
    assert_ne!(
        rbDestructible_LaunchImpactor(w, id, &bad, &mut shot_index),
        RbResult::RB_OK,
        "an impactor needs a mass"
    );
    let (mut broken, mut splits, mut pieces) = (0, 0, 0);
    for n in 0..120 {
        ok(rbWorld_Step(w));
        let mut f: RbFrame = std::mem::zeroed();
        ok(rbWorld_Poll(w, true, true, &mut f));
        if n == 0 {
            // The shot is a body the host can draw, moving as launched.
            let mut ball = 0;
            ok(rbDestructible_GetImpactorBody(w, id, 0, &mut ball));
            assert_ne!(ball, 0);
            let mut t = RbTransform::default();
            ok(rbBody_GetTransform(w, ball, &mut t));
            // Within one step of its launch point (cm, at the launch speed).
            let (at, speed) = (unreal.position_out(origin), unreal.linear_out(v));
            let d = ((t.p.x - at[0]).powi(2) + (t.p.y - at[1]).powi(2) + (t.p.z - at[2]).powi(2))
                .sqrt();
            let step = (speed[0].powi(2) + speed[1].powi(2) + speed[2].powi(2)).sqrt() / 60.0;
            assert!(
                d < 1.5 * step + 1.0,
                "the shot is {d} cm from its launch point after a step of {step} cm"
            );
        }
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
                        let bodies = std::slice::from_raw_parts(c.bodies, c.chunk_count as usize);
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
fn stress_destructibles_break_and_split_through_the_abi() {
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
            // A tenth of the masonry's strength, as the lab's slider sets it: at full
            // strength the arch holds the lab's shot (in the lab too).
            0.1,
        );
        assert!(arch.0 > 0 && arch.1 > 0 && arch.2 > 0, "arch: {arch:?}");

        let (lab, _) = engine_fracture::samples::cuboid_stress_scene(None).unwrap();
        let house = run(
            |w, out| ok(rbCreateDestructibleAssetCuboidHouse(w, out)),
            &lab,
            lab.center,
            1.0,
        );
        assert!(
            house.0 > 0 && house.1 > 0 && house.2 > 0,
            "house: {house:?}"
        );
        eprintln!("arch {arch:?}, house {house:?} (bonds broken, splits, pieces)");
    }
}
