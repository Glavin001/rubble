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
    if std::env::var("GRID_CELL").is_ok() {
        eprintln!("grid_cell_size={:.4}", world.params().grid_cell_size);
    }
    let bodies = world.download_bodies();
    let contacts = world.download_contacts();
    let mut ball_i = 0usize;
    let mut best_v = 0.0f32;
    let mut flying = 0u32;
    let mut below = 0u32;
    let mut n = 0u32;
    let mut brick_ke = 0.0f32;
    let mut n_sleeping = 0u32;
    for (i, b) in bodies.iter().enumerate() {
        if !b.is_dynamic() {
            continue;
        }
        n += 1;
        if b.is_sleeping() { n_sleeping += 1; }
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
        "steps={steps} awake={} sleep_ev={} contacts={} manifolds={} cand={} below1m={below}/{n} flying(v>1)={flying} ball_z={:.2} vz={:.2} ball_ke={ball_ke:.0} total_ke={:.0} sleeping={n_sleeping}",
        c.awake_count, c.sleep_events,
        c.contact_count,
        c.manifold_count,
        c.candidate_count,
        ball.position[2],
        ball.linear_vel[2],
        brick_ke
    );

    if std::env::var("FLYERS").is_ok() {
        let mut n = 0u32;
        for (i, b) in bodies.iter().enumerate() {
            if !b.is_dynamic() { continue; }
            let v = glam::Vec3::from_array(b.linear_vel).length();
            if v > 1.0 {
                println!(
                    "flyer i={i} pos={:.2?} v={:.3} y={:.2}",
                    b.position, v, b.position[1]
                );
                n += 1;
                if n >= 15 { break; }
            }
        }
    }
    if std::env::var("RESTLESS").is_ok() {
        let thr: f32 = std::env::var("RESTLESS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.25);
        let mut near = 0u32;
        let mut far = 0u32;
        for b in &bodies {
            if !b.is_dynamic() { continue; }
            let v = glam::Vec3::from_array(b.linear_vel).length();
            if v <= thr { continue; }
            let r = (b.position[0] * b.position[0] + b.position[2] * b.position[2]).sqrt();
            if r < 25.0 { near += 1; } else { far += 1; }
        }
        println!("restless(v>{thr}): near_xz25={near} far={far}");
        let mut printed = 0u32;
        for (i, b) in bodies.iter().enumerate() {
            if !b.is_dynamic() { continue; }
            let v = glam::Vec3::from_array(b.linear_vel).length();
            if v <= thr { continue; }
            let r = (b.position[0] * b.position[0] + b.position[2] * b.position[2]).sqrt();
            if r < 25.0 {
                let w = glam::Vec3::from_array(b.angular_vel).length();
                println!("near_restless i={i} pos={:.2?} v={v:.3} w={w:.3} y={:.2}", b.position, b.position[1]);
                printed += 1;
                if printed >= 10 { break; }
            }
        }
    }


    if std::env::var("C0_STATS").is_ok() {
        let contacts = world.download_contacts();
        let mut n = 0u32;
        let mut sum = 0.0f64;
        let mut min_c0 = f32::INFINITY;
        let mut max_c0 = f32::NEG_INFINITY;
        let mut neg = 0u32;
        for c in &contacts {
            if c.alive == 0 { continue; }
            let c0 = c.c0_n;
            n += 1;
            sum += c0 as f64;
            min_c0 = min_c0.min(c0);
            max_c0 = max_c0.max(c0);
            if c0 < 0.0 { neg += 1; }
        }
        println!("C0_STATS n={n} mean={:.6} min={min_c0:.4} max={max_c0:.4} neg={neg}", sum / n.max(1) as f64);
    }
    if std::env::var("REST_SPEED_STATS").is_ok() {
        let mut over = 0u32;
        let mut sum_v = 0.0f64;
        let mut n = 0u32;
        for b in &bodies {
            if !b.is_dynamic() { continue; }
            let v = glam::Vec3::from_array(b.linear_vel).length();
            let w = glam::Vec3::from_array(b.angular_vel).length();
            let slow = v + w * 0.5;
            n += 1;
            sum_v += slow as f64;
            if slow >= 0.1 { over += 1; }
        }
        println!("REST_SPEED_STATS n={n} mean_slow={:.4} over_0.1={over}", sum_v / n.max(1) as f64);
    }

    // Classify contacts: plane vs box-box, vertical vs horizontal normals.
    let colliders = unsafe {
        std::slice::from_raw_parts(
            world.colliders_ptr(),
            world.collider_count() as usize,
        )
    };
        if std::env::var("LONG_MF").is_ok() {
        let mut long = 0u32;
        let mut max_d = 0.0f32;
        let mut dyn_edges = 0u32;
        let mut block = 0u32;
        for m in world.download_manifolds() {
            if m.contact_count == 0 { continue; }
            let ba = colliders[m.collider_a as usize].body as usize;
            let bb = colliders[m.collider_b as usize].body as usize;
            if ba >= bodies.len() || bb >= bodies.len() { continue; }
            let a = &bodies[ba];
            let b = &bodies[bb];
            if !a.is_dynamic() || !b.is_dynamic() { continue; }
            dyn_edges += 1;
            let pa = glam::Vec3::from_array(a.position);
            let pb = glam::Vec3::from_array(b.position);
            let d = pa.distance(pb);
            max_d = max_d.max(d);
            let va = glam::Vec3::from_array(a.linear_vel).length();
            let vb = glam::Vec3::from_array(b.linear_vel).length();
            let thr = world.params().sleep_lin;
            if va > thr || vb > thr { block += 1; }
            if d > 5.0 {
                long += 1;
                if long <= 8 {
                    println!("long_dyn d={d:.2} a={ba}(v={va:.2}) b={bb}(v={vb:.2})");
                }
            }
        }
        println!("dyn_manifolds={dyn_edges} blocking_edges(v>sleep)={block} long_dyn(d>5)={long} max_d={max_d:.2}");
    }
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
        let margin = world.params().contact_offset;
        let deep = c0s.iter().filter(|&&c| c < -0.05).count();
        let deep2 = c0s.iter().filter(|&&c| c < -0.2).count();
        // Geometric penetration depth along n (negative = overlap), independent of margin in C0.
        let geoms: Vec<f32> = c0s.iter().map(|&c| c - margin).collect();
        let gdeep = geoms.iter().filter(|&&g| g < -0.05).count();
        let gdeep2 = geoms.iter().filter(|&&g| g < -0.2).count();
        println!(
            "kN: p50={:.1} p90={:.1} p99={:.1} max={:.1} | c0N: p50={:.5} p90={:.5} p01={:.5} deep<-0.05={} deep<-0.2={} | geom(c0-m): p01={:.5} deep<-0.05={} deep<-0.2={} (m={:.3})",
            pct(&kns,0.5), pct(&kns,0.9), pct(&kns,0.99), kns.last().copied().unwrap_or(0.0),
            pct(&c0s,0.5), pct(&c0s,0.9), pct(&c0s,0.01), deep, deep2,
            pct(&geoms,0.01), gdeep, gdeep2, margin
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



    if std::env::var("AABB_OV").is_ok() {
        let bodies = world.download_bodies();
        let mut pen_gt_aabb = 0u32;
        let mut n = 0u32;
        for ct in world.download_contacts() {
            if ct.alive == 0 || ct.c0_n >= -0.05 { continue; }
            let ca = ct.collider_a as usize;
            let cb = ct.collider_b as usize;
            if colliders[ca].shape != ShapeType::Box as u32 || colliders[cb].shape != ShapeType::Box as u32 {
                continue;
            }
            let ba = &bodies[colliders[ca].body as usize];
            let bb = &bodies[colliders[cb].body as usize];
            let ha = glam::Vec3::new(colliders[ca].params[0], colliders[ca].params[1], colliders[ca].params[2]);
            let hb = glam::Vec3::new(colliders[cb].params[0], colliders[cb].params[1], colliders[cb].params[2]);
            // Conservative: sphere overlap depth
            let ra = ha.length();
            let rb = hb.length();
            let d = (glam::Vec3::from_array(ba.position) - glam::Vec3::from_array(bb.position)).length();
            let sphere_pen = (ra + rb) - d;
            n += 1;
            if -ct.c0_n > sphere_pen + 0.05 {
                pen_gt_aabb += 1;
            }
        }
        println!("aabb_ov: deep_box={n} c0_deeper_than_sphere={pen_gt_aabb}");
    }
    if std::env::var("COLOR_CHECK").is_ok() {
        let colors = world.download_body_colors();
        let manifolds = world.download_manifolds();
        let mut same = 0u32;
        let mut edges = 0u32;
        for m in &manifolds {
            if m.contact_count == 0 {
                continue;
            }
            let ba = colliders[m.collider_a as usize].body as usize;
            let bb = colliders[m.collider_b as usize].body as usize;
            if ba >= colors.len() || bb >= colors.len() {
                continue;
            }
            let ca = colors[ba];
            let cb = colors[bb];
            if ca == 0xffffffff || cb == 0xffffffff {
                continue;
            }
            // Only dynamic-dynamic edges matter for Jacobi races.
            if !bodies[ba].is_dynamic() || !bodies[bb].is_dynamic() {
                continue;
            }
            edges += 1;
            if ca == cb {
                same += 1;
            }
        }
        println!(
            "color_check: dyn_edges={edges} same_color={same} color_count={}",
            world.last_counters().color_count
        );
    }

    if std::env::var("DEEP_CLASS").is_ok() {
        let bodies = world.download_bodies();
        let mut plane_deep = 0u32;
        let mut box_deep = 0u32;
        let mut geom_vs_c0_err = 0u32;
        let mut max_err = 0.0f32;
        for ct in world.download_contacts() {
            if ct.alive == 0 || ct.c0_n >= -0.05 { continue; }
            let ca = ct.collider_a as usize;
            let cb = ct.collider_b as usize;
            let sa = colliders.get(ca).map(|c| c.shape).unwrap_or(0);
            let sb = colliders.get(cb).map(|c| c.shape).unwrap_or(0);
            let plane = sa == ShapeType::Plane as u32 || sb == ShapeType::Plane as u32;
            if plane { plane_deep += 1; } else { box_deep += 1; }
            let ba = bodies[colliders[ca].body as usize];
            let bb = bodies[colliders[cb].body as usize];
            let qa = glam::Quat::from_array(ba.orientation);
            let qb = glam::Quat::from_array(bb.orientation);
            let wa = glam::Vec3::from_array(ba.position) + qa * glam::Vec3::from_array(ct.anchor_a);
            let wb = glam::Vec3::from_array(bb.position) + qb * glam::Vec3::from_array(ct.anchor_b);
            let n = glam::Vec3::from_array(ct.normal);
            let geom = n.dot(wb - wa) + 0.01; // margin
            let err = (geom - ct.c0_n).abs();
            if err > 0.05 { geom_vs_c0_err += 1; }
            max_err = max_err.max(err);
        }
        println!(
            "deep_class: plane={plane_deep} box={box_deep} geom_vs_c0_err(>5cm)={geom_vs_c0_err} max_err={max_err:.3}"
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
