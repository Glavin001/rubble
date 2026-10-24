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
        dd.start_asleep = true;
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
        let step = |w: *mut RbWorld, n: u32, label: &str| {
            let t = Instant::now();
            let mut moves = 0;
            let mut events = 0;
            for _ in 0..n {
                ok(rbWorld_Step(w));
                let mut f = std::mem::zeroed::<RbFrame>();
                ok(rbWorld_Poll(w, true, true, &mut f));
                moves = moves.max(f.move_count);
                events += f.destruction_count;
            }
            println!("{label}: {n} steps, {:.2} ms/step, max moves {moves}, destruction events {events}", t.elapsed().as_secs_f64() * 1e3 / n as f64);
        };
        step(w, 120, "at rest");
        // 2 t balls at 40 m/s square into Elm Park cottage walls (cottage-82's south wall at
        // x -72.6, z 18.7, and its neighbours').
        for x in [-72.6f32, -70.0, -75.0] {
            let mut bd = rbDefaultBodyDef();
            bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
            bd.transform.p = RbVec3 { x, y: 2.5, z: 40.0 };
            bd.transform.q.w = 1.0;
            bd.linear_velocity = RbVec3 { x: 0.0, y: 0.0, z: -40.0 };
            let mut ball = 0;
            ok(rbCreateBody(w, &bd, &mut ball));
            let mut sd = rbDefaultShapeDef();
            sd.density = 4000.0;
            ok(rbCreateSphereShape(w, ball, &sd, RbVec3::default(), 0.6, &mut s));
        }
        step(w, 120, "shots in flight");
        step(w, 240, "after the shots");
        rbDestroyWorld(w);
    }
}
