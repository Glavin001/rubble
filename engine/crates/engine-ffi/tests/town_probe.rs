//! Probe (ignored): can a world carry Vibe Town's ScenePack (52k chunks) as one destructible?
//! Options: TOWN_UNREAL (Unreal frame and units), TOWN_LAB_SOLVER, TOWN_THRESHOLD (the threshold
//! strength model and its solver), TOWN_BULLET, TOWN_TRACE (fastest chunk per step).
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
        def.max_contacts = if std::env::var("TOWN_THRESHOLD").is_ok() { 2_000_000 } else { 1 << 21 };
        def.max_candidates = 1 << 21;
        def.max_joints = 1 << 18;
        def.max_hull_vertices = 1 << 21;
        let threshold = std::env::var("TOWN_THRESHOLD").is_ok();
        let columns: Option<f32> = std::env::var("TOWN_COLUMNS").ok().map(|v| v.parse().unwrap_or(45.0));
        if threshold {
            def.contacts = RbContacts::RB_CONTACTS_CANONICAL;
        }
        // TOWN_UNREAL: the world in Unreal's frame and units (Z up, cm), as the Unreal client runs it.
        let unreal = std::env::var("TOWN_UNREAL").is_ok();
        // Rubble's frame (Y up, m) to the world's.
        let to_w = |v: RbVec3, len: f32| if unreal { RbVec3 { x: -v.z * len, y: v.x * len, z: v.y * len } } else { v };
        if unreal {
            def.coordinates = rubble_ffi::coords::RbCoordinates::RB_COORDINATES_UNREAL;
            def.gravity = RbVec3 { x: 0.0, y: 0.0, z: -981.0 };
        }
        let mut w = ptr::null_mut();
        ok(rbCreateWorld(&def, &mut w));
        let mut g = 0;
        ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
        let mut s = 0;
        ok(rbCreatePlaneShape(w, g, &rbDefaultShapeDef(), to_w(RbVec3 { x: 0.0, y: 1.0, z: 0.0 }, 1.0), 0.0, &mut s));
        let t = Instant::now();
        let mut asset = 0;
        ok(rbCreateDestructibleAssetFromPack(w, bytes.as_ptr(), bytes.len(), &mut asset));
        if threshold {
            let mut dt = 0.0f32;
            ok(rbWorld_UseDestructibleAssetSolverFor(w, asset, RbStrengthModel::RB_STRENGTH_THRESHOLD, &mut dt));
            let (mut has, mut at) = (false, RbVec3::default());
            ok(rbWorld_GetDestructibleAssetGround(w, asset, &mut has, &mut at));
            println!("threshold solver: step {dt} s, ground {has} at {:?}", (at.x, at.y, at.z));
        } else if std::env::var("TOWN_LAB_SOLVER").is_ok() {
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
        if threshold {
            dd.model = RbStrengthModel::RB_STRENGTH_THRESHOLD;
        }
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
        assert!(local.p.x.abs() + local.p.y.abs() + local.p.z.abs() > 1.0, "chunk {} local {:?}", chunks / 2, local.p);
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
        // Every chunk's body and pose, for where its centre is (body x pose).
        let mut bodies = vec![0; chunks as usize];
        let mut poses = vec![RbTransform::default(); chunks as usize];
        for c in 0..chunks {
            ok(rbDestructible_GetChunkBody(w, d, c, &mut bodies[c as usize]));
            ok(rbDestructible_GetChunkPose(w, d, c, &mut poses[c as usize]));
        }
        let up = if unreal { 2 } else { 1 };
        let unit = if unreal { 100.0 } else { 1.0 };
        let centres = |w: *mut RbWorld| -> Vec<[f32; 3]> {
            let mut t = vec![RbTransform::default(); bodies.len()];
            ok(rbWorld_GetTransforms(w, bodies.as_ptr(), bodies.len() as u32, t.as_mut_ptr()));
            t.iter().zip(&poses).map(|(t, l)| {
                let (q, v) = ([t.q.x, t.q.y, t.q.z], [l.p.x, l.p.y, l.p.z]);
                let cr = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
                let u = cr(q, v);
                let u2 = cr(q, u);
                let p = [t.p.x, t.p.y, t.p.z];
                [0, 1, 2].map(|k| p[k] + v[k] + 2.0 * (t.q.w * u[k] + u2[k]))
            }).collect()
        };
        let start = centres(w);
        // Tower-187's roof: its highest chunk, re-read through its current body (splits move it).
        // In Rubble's frame: (x, z, up), metres.
        let rubble_xzy = |p: RbVec3| if unreal { [p.y / 100.0, -p.x / 100.0, p.z / 100.0] } else { [p.x, p.z, p.y] };
        let roof = (0..chunks)
            .filter(|&c| {
                let [x, z, _] = rubble_xzy(poses[c as usize].p);
                (83.0..97.5).contains(&x) && (55.5..70.0).contains(&z)
            })
            .max_by(|&a, &b| rubble_xzy(poses[a as usize].p)[2].total_cmp(&rubble_xzy(poses[b as usize].p)[2]));
        let roof_height = |w: *mut RbWorld| -> f32 {
            let c = roof.unwrap();
            let mut b = 0;
            ok(rbDestructible_GetChunkBody(w, d, c, &mut b));
            let mut t = RbTransform::default();
            ok(rbBody_GetTransform(w, b, &mut t));
            let l = poses[c as usize].p;
            let (q, v) = ([t.q.x, t.q.y, t.q.z], [l.x, l.y, l.z]);
            let cr = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
            let u = cr(q, v);
            let u2 = cr(q, u);
            let p = [t.p.x, t.p.y, t.p.z];
            (p[up] + v[up] + 2.0 * (t.q.w * u[up] + u2[up])) / unit
        };
        let report = |w: *mut RbWorld, label: &str| {
            let now = centres(w);
            let (mut moved, mut far, mut rose, mut top) = (0, 0, 0, 0.0f32);
            for (a, b) in start.iter().zip(&now) {
                let dd = ((0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>()).sqrt() / unit;
                moved += (dd > 0.1) as u32;
                far += (dd > 5.0) as u32;
                let r = (b[up] - a[up]) / unit;
                rose += (r > 2.0) as u32;
                top = top.max(r);
            }
            if columns.is_some() {
                println!("  {label}: tower-187 roof at {:.1} m", roof_height(w));
            }
            println!("  {label}: chunks moved >0.1 m {moved}, >5 m {far}; risen >2 m {rose}, most risen {top:.1} m");
        };
        step(w, 120, "at rest");
        report(w, "at rest");
        // The Unreal test's hits: a 7 t ball at 45 m/s into house-85's wall, then a 22 t meteor
        // straight down onto house-89 at 50 m/s.
        let mut shoot = |p: RbVec3, v: RbVec3, r: f32, density: f32| {
            let mut bd = rbDefaultBodyDef();
            bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
            let u = if unreal { 100.0 } else { 1.0 };
            bd.transform.p = to_w(p, u);
            bd.transform.q.w = 1.0;
            bd.linear_velocity = to_w(v, u);
            bd.is_bullet = std::env::var("TOWN_BULLET").is_ok();
            let mut ball = 0;
            ok(rbCreateBody(w, &bd, &mut ball));
            let mut sd = rbDefaultShapeDef();
            sd.density = if unreal { density * 1e-6 } else { density };
            ok(rbCreateSphereShape(w, ball, &sd, RbVec3::default(), r * u, &mut s));
            if threshold {
                ok(rbBody_SetProjectile(w, ball, &rbDefaultProjectile()));
            }
            ball
        };
        // TOWN_COLUMNS: tower-187's four column rows shot from the side (-x) at column height,
        // one ball a row (TOWN_COLUMNS=speed m/s, default 45); else the house shots.
        let mut fired = Vec::new();
        if let Some(speed) = columns {
            let lift = if threshold { 0.5 } else { 0.0 };
            for z in [56.9f32, 60.9, 64.9, 68.9] {
                fired.push(shoot(RbVec3 { x: 76.0, y: 1.3 + 0.5 + lift, z: z - 0.365 }, RbVec3 { x: speed, y: 0.0, z: 0.0 }, 0.6, 7800.0));
            }
        }
        let shots = if columns.is_some() { [fired[0], fired[1]] } else { [shoot(RbVec3 { x: -51.2, y: 2.5, z: 40.0 }, RbVec3 { x: 0.0, y: 0.0, z: -45.0 }, 0.6, 7800.0),
        shoot(RbVec3 { x: -34.2, y: 50.0, z: 15.5 }, RbVec3 { x: 0.0, y: -50.0, z: 0.0 }, 1.2, 3000.0)] };
        // TOWN_TRACE: per step, the fastest chunk (from its centre's move) once one passes 25 m/s.
        if std::env::var("TOWN_TRACE").is_ok() {
            let mut prev = centres(w);
            let mut broken = 0u32;
            for n in 0..600 {
                ok(rbWorld_Step(w));
                let mut f = std::mem::zeroed::<RbFrame>();
                ok(rbWorld_Poll(w, true, true, &mut f));
                let mut nb = 0;
                for e in std::slice::from_raw_parts(f.destruction, f.destruction_count as usize) {
                    if e.type_ == RbDestructionType::RB_BONDS_BROKEN {
                        nb += e.bond_count;
                    }
                }
                broken += nb;
                let now = centres(w);
                let (mut best, mut at) = (0.0f32, 0usize);
                let mut fast = 0;
                for (k, (a, b)) in prev.iter().zip(&now).enumerate() {
                    let v = ((0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>()).sqrt() / unit * 60.0;
                    fast += (v > 25.0) as u32;
                    if v > best {
                        best = v;
                        at = k;
                    }
                }
                if n % 15 == 0 && n < 120 {
                    let mut t = [RbTransform::default(); 2];
                    ok(rbWorld_GetTransforms(w, shots.as_ptr(), 2, t.as_mut_ptr()));
                    println!("step {n}: ball {:?}, meteor {:?}, bonds broken so far {broken}", [t[0].p.x, t[0].p.y, t[0].p.z].map(|x| x / unit), [t[1].p.x, t[1].p.y, t[1].p.z].map(|x| x / unit));
                }
                if best > 25.0 {
                    let c = now[at];
                    println!("step {n}: fastest chunk {at} at {best:.0} m/s, centre {:?}, {fast} chunks over 25 m/s, bonds broken this step {nb} (total {broken})", c.map(|x| x / unit));
                    if fast > 50 { break; }
                }
                prev = now;
            }
            rbDestroyWorld(w);
            return;
        }
        for second in 0..10 {
            step(w, 60, &format!("second {second}"));
            report(w, &format!("second {second}"));
        }

        rbDestroyWorld(w);
    }
}
