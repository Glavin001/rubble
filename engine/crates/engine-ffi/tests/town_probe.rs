//! Probe (ignored): can a world carry Vibe Town's ScenePack (52k chunks) as one destructible?
//! RUBBLE_TOWN_PACK=/path/vibe-town.json cargo test -p engine-ffi --release --test town_probe -- --ignored --nocapture
use rubble_ffi::*;
use std::ptr;
use std::time::Instant;

fn ok(r: RbResult) {
    assert_eq!(r, RbResult::RB_OK, "{:?}", unsafe { std::ffi::CStr::from_ptr(rbGetLastError()) });
}

#[test]
#[ignore]
fn town_pack_loads_steps_and_breaks() {
    let Ok(path) = std::env::var("RUBBLE_TOWN_PACK") else { return };
    let bytes = std::fs::read(&path).unwrap();
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.max_bodies = 1 << 17;
        def.max_colliders = 1 << 18;
        def.max_contacts = 1 << 21;
        def.max_candidates = 1 << 21;
        def.max_joints = 1 << 18;
        def.max_hull_vertices = 1 << 21;
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let mut g = 0;
        ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
        let mut s = 0;
        ok(rbCreatePlaneShape(w, g, &rbDefaultShapeDef(), RbVec3 { x: 0.0, y: 1.0, z: 0.0 }, 0.0, &mut s));
        let t = Instant::now();
        let mut asset = 0;
        ok(rbCreateDestructibleAssetFromPack(w, bytes.as_ptr(), bytes.len(), &mut asset));
        if std::env::var("TOWN_LAB_SOLVER").is_ok() {
            let mut dt = 0.0f32;
            ok(rbWorld_UseDestructibleAssetSolver(w, asset, &mut dt));
            println!("lab solver: step {dt} s");
        }
        if std::env::var("TOWN_PRELOAD").is_ok() {
            let t = Instant::now();
            let mut settled = false;
            ok(rbWorld_PreloadDestructibleAsset(w, asset, &mut settled));
            println!("preload: settled={settled} in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        let mut dd = rbDefaultDestructibleDef();
        dd.asset = asset;
        dd.start_asleep = std::env::var("TOWN_AWAKE").is_err();
        dd.anchored = true;
        println!("default strength_scale {}", dd.strength_scale);
        if let Ok(v) = std::env::var("TOWN_STRENGTH") {
            dd.strength_scale = v.parse().unwrap();
        }
        let mut d = 0;
        ok(rbCreateDestructible(w, &dd, &mut d));
        let mut chunks = 0;
        ok(rbDestructible_GetChunkCount(w, d, &mut chunks));
        println!("loaded {chunks} chunks in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        for c in [0, chunks / 2, chunks - 1] {
            let mut node = -1;
            ok(rbDestructible_GetChunkNode(w, d, c, &mut node));
            assert_eq!(node, c as i32, "chunk {c} is pack node {c}");
        }
        // Chunk bodies start at the destructible's frame; each chunk sits at its pose.
        let (mut b, mut t, mut local) = (0, RbTransform::default(), RbTransform::default());
        ok(rbDestructible_GetChunkBody(w, d, chunks / 2, &mut b));
        ok(rbBody_GetTransform(w, b, &mut t));
        ok(rbDestructible_GetChunkPose(w, d, chunks / 2, &mut local));
        if std::env::var("TOWN_PRELOAD").is_err() {
            // (A preload starts each body at its settled pose instead.)
            assert!(t.p.x.abs() + t.p.y.abs() + t.p.z.abs() < 1e-6, "chunk body at the destructible's frame: {:?}", t.p);
        }
        assert!(local.p.x.abs() + local.p.z.abs() > 1.0, "chunk {} local {:?}", chunks / 2, local.p);
        let step = |w: *mut RbWorld, n: u32, label: &str| {
            let t = Instant::now();
            let mut moves = 0;
            let mut events = 0;
            let mut bonds = 0u32;
            for _ in 0..n {
                ok(rbWorld_Step(w));
                let mut f = std::mem::zeroed::<RbFrame>();
                ok(rbWorld_Poll(w, true, true, &mut f));
                moves = moves.max(f.move_count);
                events += f.destruction_count;
                for e in std::slice::from_raw_parts(f.destruction, f.destruction_count as usize) {
                    if e.type_ == RbDestructionType::RB_BONDS_BROKEN {
                        bonds += e.bond_count;
                    }
                }
            }
            println!("{label}: {n} steps, {:.2} ms/step, max moves {moves}, destruction events {events}, bonds broken {bonds}", t.elapsed().as_secs_f64() * 1e3 / n as f64);
        };
        step(w, 120, "at rest");
        // The Unreal test's hits: a 7 t ball at 45 m/s into house-85's wall, then a 22 t meteor
        // straight down onto house-89 at 50 m/s.
        let mut shoot = |p: RbVec3, v: RbVec3, r: f32, density: f32| {
            let mut bd = rbDefaultBodyDef();
            bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
            bd.transform.p = p;
            bd.transform.q.w = 1.0;
            bd.linear_velocity = v;
            let mut ball = 0;
            ok(rbCreateBody(w, &bd, &mut ball));
            let mut sd = rbDefaultShapeDef();
            sd.density = density;
            ok(rbCreateSphereShape(w, ball, &sd, RbVec3::default(), r, &mut s));
        };
        shoot(RbVec3 { x: -51.2, y: 2.5, z: 40.0 }, RbVec3 { x: 0.0, y: 0.0, z: -45.0 }, 0.6, 7800.0);
        shoot(RbVec3 { x: -34.2, y: 50.0, z: 15.5 }, RbVec3 { x: 0.0, y: -50.0, z: 0.0 }, 1.2, 3000.0);
        for second in 0..10 {
            step(w, 60, &format!("second {second}"));
        }

        rbDestroyWorld(w);
    }
}
