use engine_exec::PhysicsBackend;
use engine_schema::ShapeType;

fn main() {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Brick Ring (9k)".into());
    let steps: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let (mut world, _) = engine_scenes::build_world(&name).expect("build");
    for _ in 0..steps {
        world.step().expect("step");
    }
    let c = world.last_counters();
    let bodies = world.download_bodies();
    let contacts = world.download_contacts();
    let mut ball_i = 0usize;
    let mut best_v = 0.0f32;
    let mut flying = 0u32;
    let mut below = 0u32;
    let mut n = 0u32;
    let mut brick_ke = 0.0f32;
    for (i, b) in bodies.iter().enumerate() {
        if !b.is_dynamic() {
            continue;
        }
        n += 1;
        let v = glam::Vec3::from_array(b.linear_vel).length();
        if v > best_v {
            best_v = v;
            ball_i = i;
        }
        if b.position[1] < 1.0 {
            below += 1;
        }
        if v > 1.0 {
            flying += 1;
        }
        let m = if b.inv_mass > 1e-12 {
            1.0 / b.inv_mass
        } else {
            0.0
        };
        brick_ke += 0.5 * m * v * v;
    }
    let ball = &bodies[ball_i];
    let bm = if ball.inv_mass > 1e-12 {
        1.0 / ball.inv_mass
    } else {
        0.0
    };
    let ball_ke = 0.5 * bm * glam::Vec3::from_array(ball.linear_vel).length_squared();
    println!(
        "steps={steps} awake={} contacts={} manifolds={} cand={} below1m={below}/{n} flying(v>1)={flying} ball_z={:.2} vz={:.2} ball_ke={ball_ke:.0} total_ke={:.0}",
        c.awake_count,
        c.contact_count,
        c.manifold_count,
        c.candidate_count,
        ball.position[2],
        ball.linear_vel[2],
        brick_ke
    );

    // Classify contacts: plane vs box-box, vertical vs horizontal normals.
    let colliders = unsafe {
        std::slice::from_raw_parts(
            world.colliders_ptr(),
            world.collider_count() as usize,
        )
    };
    let mut plane_c = 0u32;
    let mut box_c = 0u32;
    let mut other_c = 0u32;
    let mut vert = 0u32; // |ny| > 0.85
    let mut horiz = 0u32;
    let mut pairs = std::collections::HashSet::new();
    let mut pts_hist = [0u32; 9];
    let mut pair_pts: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for ct in &contacts {
        if ct.alive == 0 {
            continue;
        }
        let ca = ct.collider_a as usize;
        let cb = ct.collider_b as usize;
        let sa = colliders.get(ca).map(|c| c.shape).unwrap_or(0);
        let sb = colliders.get(cb).map(|c| c.shape).unwrap_or(0);
        let plane = sa == ShapeType::Plane as u32 || sb == ShapeType::Plane as u32;
        if plane {
            plane_c += 1;
        } else if sa == ShapeType::Box as u32 && sb == ShapeType::Box as u32 {
            box_c += 1;
        } else {
            other_c += 1;
        }
        if ct.normal[1].abs() > 0.85 {
            vert += 1;
        } else {
            horiz += 1;
        }
        let key = (ct.collider_a.min(ct.collider_b), ct.collider_a.max(ct.collider_b));
        pairs.insert(key);
        *pair_pts.entry(key).or_insert(0) += 1;
    }
    for (_k, pts) in &pair_pts {
        let i = (*pts as usize).min(8);
        pts_hist[i] += 1;
    }
    println!(
        "classify: plane={plane_c} box-box={box_c} other={other_c} vert(|ny|>0.85)={vert} horiz={horiz} unique_pairs={}",
        pairs.len()
    );
    println!(
        "pts/pair hist: 1={} 2={} 3={} 4={} 5+={}",
        pts_hist[1],
        pts_hist[2],
        pts_hist[3],
        pts_hist[4],
        pts_hist[5] + pts_hist[6] + pts_hist[7] + pts_hist[8]
    );

    // Warm-start key survival over the last two steps (churn diagnosis).
    {
        let mut prev: std::collections::HashSet<(u32, u32, u32)> = std::collections::HashSet::new();
        for ct in &contacts {
            if ct.alive != 0 {
                prev.insert((ct.collider_a.min(ct.collider_b), ct.collider_a.max(ct.collider_b), ct.feature_id));
            }
        }
        world.step().expect("step");
        let cur = world.download_contacts();
        let mut hit = 0u32;
        let mut n = 0u32;
        for ct in &cur {
            if ct.alive == 0 { continue; }
            n += 1;
            let k = (ct.collider_a.min(ct.collider_b), ct.collider_a.max(ct.collider_b), ct.feature_id);
            if prev.contains(&k) { hit += 1; }
        }
        let surv = if n > 0 { hit as f32 / n as f32 } else { 0.0 };
        println!("warmstart_survival={surv:.3} ({hit}/{n})");
        let reuse = world.scratch_u32(18);
        let miss = world.scratch_u32(19);
        let gen_reject = world.scratch_u32(20);
        let cand = world.last_counters().candidate_count;
        let rate = if cand > 0 {
            reuse as f32 / cand as f32
        } else {
            0.0
        };
        println!(
            "reuseContacts_hits={reuse}/{cand} ({rate:.3}) hash_miss={miss} gen_reject={gen_reject} refs_noRef={} lin={} ang={}",
            world.scratch_u32(21),
            world.scratch_u32(22),
            world.scratch_u32(23),
        );
    }

    // Velocity percentiles (VEL_HIST=1)

    if std::env::var("KN_HIST").is_ok() {
        let mut kns: Vec<f32> = Vec::new();
        let mut c0s: Vec<f32> = Vec::new();
        for ct in world.download_contacts() {
            if ct.alive == 0 { continue; }
            kns.push(ct.k_n);
            c0s.push(ct.c0_n);
        }
        kns.sort_by(|a,b| a.partial_cmp(b).unwrap());
        c0s.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let pct = |v: &[f32], p: f32| -> f32 {
            if v.is_empty() { return 0.0; }
            let i = ((v.len() as f32 - 1.0) * p).round() as usize;
            v[i.min(v.len()-1)]
        };
        println!(
            "kN: p50={:.1} p90={:.1} p99={:.1} max={:.1} | c0N: p50={:.5} p90={:.5} p01={:.5}",
            pct(&kns,0.5), pct(&kns,0.9), pct(&kns,0.99), kns.last().copied().unwrap_or(0.0),
            pct(&c0s,0.5), pct(&c0s,0.9), pct(&c0s,0.01)
        );
    }
    if std::env::var("GEN_HIST").is_ok() {
        let step = world.scratch_u32(17);
        let bodies2 = world.download_bodies();
        let mut gens: Vec<u32> = bodies2
            .iter()
            .filter(|b| b.is_dynamic())
            .map(|b| b.generation)
            .collect();
        let n = gens.len() as u32;
        let at_step = gens.iter().filter(|&&g| g == step).count() as u32;
        gens.sort();
        let gp = |p: f32| -> u32 {
            if gens.is_empty() {
                return 0;
            }
            let i = ((gens.len() as f32 - 1.0) * p).round() as usize;
            gens[i.min(gens.len() - 1)]
        };
        let mut cg: Vec<u32> = world
            .download_contacts()
            .iter()
            .filter(|c| c.alive != 0)
            .map(|c| c._pad[1].to_bits())
            .collect();
        cg.sort();
        let cp = |p: f32| -> u32 {
            if cg.is_empty() {
                return 0;
            }
            let i = ((cg.len() as f32 - 1.0) * p).round() as usize;
            cg[i.min(cg.len() - 1)]
        };
        println!(
            "gen: step={step} body p50={} p90={} max={} at_cur={at_step}/{n} | ct_gen p50={} p90={} max={}",
            gp(0.5),
            gp(0.9),
            gens.last().copied().unwrap_or(0),
            cp(0.5),
            cp(0.9),
            cg.last().copied().unwrap_or(0)
        );
    }

    if std::env::var("VEL_HIST").is_ok() {
        let mut lins: Vec<f32> = Vec::new();
        let mut angs: Vec<f32> = Vec::new();
        let mut timers = 0u32;
        for b in &bodies {
            if !b.is_dynamic() {
                continue;
            }
            let lv = glam::Vec3::from_array(b.linear_vel).length();
            let av = glam::Vec3::from_array(b.angular_vel).length();
            lins.push(lv);
            angs.push(av);
            if b.sleep_timer > 0.0 {
                timers += 1;
            }
        }
        lins.sort_by(|a, b| a.partial_cmp(b).unwrap());
        angs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |v: &[f32], p: f32| -> f32 {
            if v.is_empty() {
                return 0.0;
            }
            let i = ((v.len() as f32 - 1.0) * p).round() as usize;
            v[i.min(v.len() - 1)]
        };
        let under_sleep = lins.iter().filter(|&&v| v < 0.15).count();
        let under_ang = angs.iter().filter(|&&v| v < 0.15).count();
        println!(
            "vel: lin p50={:.4} p90={:.4} p99={:.4} max={:.4} |ang p50={:.4} p90={:.4} p99={:.4}| under_sleepLin={}/{} under_sleepAng={}/{} sleepTimer>0={}",
            pct(&lins, 0.5),
            pct(&lins, 0.9),
            pct(&lins, 0.99),
            lins.last().copied().unwrap_or(0.0),
            pct(&angs, 0.5),
            pct(&angs, 0.9),
            pct(&angs, 0.99),
            under_sleep,
            lins.len(),
            under_ang,
            angs.len(),
            timers
        );
    }
}
