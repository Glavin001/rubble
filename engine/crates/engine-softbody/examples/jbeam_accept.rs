//! A JBeam vehicle in the explicit solver's JBeam mode with pressure wheels
//! (`engine/docs/jbeam-semantics-spec.md` Part II), CPU reference: the census of
//! what it uses and does not simulate, at rest for `rest=` s, then into a rigid wall
//! at `kmh=` (default 56) for `crash=` s.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example jbeam_accept -- <vehicles-dir> <name> <config.pc> [rest=3] [kmh=56] [crash=1.5] [wheels=pressure|soft]
//! ```
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, StaticBox,
};
use glam::Vec3;

fn report(tag: &str, s: &ExplicitBeams, ke0: f32) {
    let ke: f32 = s
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum();
    let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
    let nan = s.nodes.iter().any(|n| !n.pos.is_finite());
    let (broken, bent) = s.damage();
    let deform = s.deform_triggered().iter().filter(|&&d| d != 0).count();
    let dead = s.tri_dead.iter().filter(|&&d| d).count();
    let mass: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let com = s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / mass;
    println!(
        "{tag}: broken {broken} bent {bent} KE {ke:.1} J (spawn {ke0:.1}) max node speed {vmax:.2} m/s \
         NaN {nan} deform groups {deform}/{} dead triangles {dead}/{} centre of mass {com:.4}",
        s.deform_state.len(),
        s.tri_dead.len()
    );
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (dir, name, pc) = (&a[0], &a[1], &a[2]);
    let opt = |k: &str| a.iter().find_map(|x| x.strip_prefix(k)).map(str::to_string);
    let f = |k: &str, d: f32| opt(k).map_or(d, |s| s.parse().unwrap());
    let (rest, kmh, crash) = (f("rest=", 3.0), f("kmh=", 56.0), f("crash=", 1.5));
    let lift = f("lift=", 0.02);
    let dir = std::path::Path::new(dir);
    let texts = engine_softbody::jbeam::read_vehicle_dir(dir, name).unwrap();
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
    let census = engine_softbody::jbeam::feature_census(&files, Some(&pcs)).unwrap();
    let unsupported = engine_softbody::jbeam::unsupported_features(&census);
    println!("unsupported physics features: {unsupported:?}");
    println!(
        "out of scope: {:?}",
        engine_softbody::jbeam::out_of_scope_features(&census)
    );
    let desc = engine_softbody::jbeam::load_jbeam(&files, Some(&pcs)).unwrap();
    let wheels = if opt("wheels=").as_deref() == Some("soft") {
        JbeamWheels::Soft(engine_softbody::truck_export::TyreDefaults::for_vehicle(
            &desc,
        ))
    } else {
        JbeamWheels::Pressure
    };
    let settings = ExplicitBeamsSettings {
        drag: opt("drag=").is_some(),
        turbulence: false,
        ..Default::default()
    };
    let spawn = |v: Vec3| {
        let (mut s, w) =
            ExplicitBeams::from_desc_jbeam_with(&desc, wheels, Vec3::ZERO, v, settings).unwrap();
        // Facing +Z (a file can model its car along another axis: the Monster
        // supercar faces −X), so `v` and the wall are along its forward axis.
        let f = engine_softbody::pack::forward_axis(&desc);
        let turn = glam::Quat::from_rotation_y(-f.x.atan2(f.z));
        for n in &mut s.nodes {
            n.pos = turn * n.pos;
        }
        let low = s
            .nodes
            .iter()
            .map(|n| n.pos.y)
            .fold(f32::INFINITY, f32::min);
        for n in &mut s.nodes {
            n.pos.y -= low - lift;
        }
        s.roll_tyres(v);
        if let Some(cap) = opt("dampcap=") {
            let cap: f32 = cap.parse().unwrap();
            for b in &mut s.beams {
                b.c = b.c.min(cap);
            }
        }
        (s, w)
    };
    let (mut s, warnings) = spawn(Vec3::ZERO);
    if opt("nofilter=").is_some() {
        for b in &mut s.beams {
            if matches!(b.kind, engine_softbody::explicit_beams::ExplicitKind::JbeamPlain | engine_softbody::explicit_beams::ExplicitKind::JbeamBounded | engine_softbody::explicit_beams::ExplicitKind::JbeamSupport) {
                b.shock = 0;
            }
        }
    }
    if let Some(mu) = opt("wheelfric=") {
        let n0 = desc.nodes.len();
        for n in &mut s.nodes[n0..] {
            n.friction = mu.parse().unwrap();
        }
    }
    if let Some(list) = opt("nocand=") {
        for w in list.split(',') {
            let i = desc.nodes.iter().position(|n| n.name == w).unwrap() as u32;
            s.candidates.retain(|&c| c != i);
        }
    }
    if opt("noself=").is_some() {
        s.candidates.clear();
    }
    for w in &warnings {
        println!("warning: {w}");
    }
    let mass: f32 = s.nodes.iter().map(|n| n.mass).sum();
    println!("forward axis {:?}; first axle {:?}", engine_softbody::pack::forward_axis(&desc), s.tyres.first().map(|t| (s.nodes[t.axle[1] as usize].pos - s.nodes[t.axle[0] as usize].pos).normalize()));
    println!(
        "{name} {pc}: {} nodes ({} desc), {} beams ({} with extensions), {} collision triangles, {} kg, {} wheels",
        s.nodes.len(),
        desc.nodes.len(),
        s.beams.len(),
        s.beam_ext.len(),
        s.triangles.len(),
        mass,
        desc.pressure_wheels.len()
    );
    if opt("spawnpairs=").is_some() {
        // Self-collision node-triangle pairs inside the contact range at spawn.
        let name = |i: u32| desc.nodes.get(i as usize).map_or("wheel".to_string(), |n| n.name.clone());
        for (t, tri) in s.triangles.iter().enumerate() {
            let g = s.triangle_self_group(t);
            let [i0, i1, i2] = tri.nodes;
            let (p0, p1, p2) = (s.nodes[i0 as usize].pos, s.nodes[i1 as usize].pos, s.nodes[i2 as usize].pos);
            let (e1, e2) = (p1 - p0, p2 - p0);
            let n = e1.cross(e2).normalize_or_zero();
            for &c in &s.candidates {
                if c == i0 || c == i1 || c == i2 || (g != 0 && s.self_groups[c as usize] == g) {
                    continue;
                }
                let q = s.nodes[c as usize].pos - p0;
                let delta = n.dot(q);
                if delta.abs() > s.collision_range {
                    continue;
                }
                // Barycentric coordinates of the projection.
                let qp = q - n * delta;
                let (d11, d12, d22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
                let (q1, q2) = (qp.dot(e1), qp.dot(e2));
                let det = d11 * d22 - d12 * d12;
                let a = (d22 * q1 - d12 * q2) / det;
                let b = (d11 * q2 - d12 * q1) / det;
                if a >= 0.0 && b >= 0.0 && a + b <= 1.0 {
                    println!(
                        "pair: node {} triangle {} {} {} delta {:.4} m",
                        name(c), name(i0), name(i1), name(i2), delta
                    );
                }
            }
        }
        return;
    }
    if let Some(w) = opt("watch=") {
        // Follow one node: position and velocity every 5 steps for 0.4 s.
        let i = match w.strip_prefix('#') {
            Some(k) => k.parse().unwrap(),
            None => desc.nodes.iter().position(|n| n.name == w).unwrap(),
        };
        let every: usize = opt("every=").map_or(5, |e| e.parse().unwrap());
        let from: usize = opt("from=").map_or(0, |e| e.parse().unwrap());
        for step in 0..from + 800 {
            s.step();
            if step >= from && step % every == 0 {
                let n = s.nodes[i];
                println!("step {step}: {w} pos {:.4?} vel {:.3?} force {:.1?} contacts {}", n.pos, n.vel, n.force, s.last_contacts);
            }
        }
        return;
    }
    if opt("breakwatch=").is_some() {
        // The first step in which a beam breaks: its records just before.
        for step in 0..4000 {
            let before = s.beams.clone();
            s.step();
            let now: Vec<usize> = (0..s.beams.len()).filter(|&j| s.beams[j].broken && !before[j].broken).collect();
            if !now.is_empty() {
                println!("step {step}: {} beams break", now.len());
                for (j, b) in s.beams.iter().enumerate().filter(|(_, b)| b.kind == engine_softbody::explicit_beams::ExplicitKind::JbeamPressureGroup) {
                    let e = s.beam_ext[s.ext_of[j] as usize - 1];
                    println!("  pressure group {j}: V0 {} p_abs0 {} ceiling {} gauge now {:?}", e.e1[2], e.e1[3], e.e2[0], e.pressure);
                    let _ = b;
                }
                for &j in now.iter().take(6) {
                    let b = before[j];
                    let (pa, pb) = (s.nodes[b.a as usize].pos, s.nodes[b.b as usize].pos);
                    println!("  beam {j} {:?} before {:?} len now {:.4} ext {:?}", b.kind, b, pa.distance(pb),
                        s.ext_of.get(j).filter(|&&x| x != 0).map(|&x| s.beam_ext[x as usize - 1]));
                }
                return;
            }
        }
        println!("no break in 2 s");
        return;
    }
    if opt("aero=").is_some() {
        // Drag area per dragCoef: Σ C·(dragCoef/100)·A·|n·ẑ| facing the car's forward axis.
        let f = engine_softbody::pack::forward_axis(&desc);
        let mut by: std::collections::BTreeMap<String, (usize, f32, f32)> = Default::default();
        for t in &desc.triangles {
            let p = t.nodes.map(|i| glam::Vec3::from_array(desc.nodes[i as usize].position));
            let cr = (p[1] - p[0]).cross(p[2] - p[0]);
            let a = 0.5 * cr.length();
            let proj = 0.5 * cr.dot(f).abs();
            let e = by.entry(format!("{:.2}/{:.2}", t.aero[0], t.aero[1])).or_default();
            e.0 += 1;
            e.1 += a;
            e.2 += 1.28 * t.aero[0] / 100.0 * proj;
        }
        let total: f32 = by.values().map(|e| e.2).sum();
        for (k, (n, a, cda)) in &by {
            println!("dragCoef/liftCoef {k}: {n} triangles, area {a:.2} m², drag area {cda:.3} m²");
        }
        println!("total drag area C·A {total:.2} m² ({} triangles)", desc.triangles.len());
        return;
    }
    if opt("modes=").is_some() {
        // The highest beam-spring mode ω_max·h (power iteration on M⁻¹K, springs only):
        // explicit integration diverges past 2.
        let n = s.nodes.len();
        let mut x: Vec<Vec3> = (0..n).map(|i| Vec3::new((i % 7) as f32 - 3.0, (i % 5) as f32 - 2.0, (i % 3) as f32 - 1.0)).collect();
        let mut lam = 0.0f32;
        for _ in 0..400 {
            let mut y = vec![Vec3::ZERO; n];
            for b in s.beams.iter().filter(|b| !b.broken && b.a != b.b && b.k > 0.0 && b.k.is_finite()) {
                let d = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).normalize_or_zero();
                let f = d * (b.k * d.dot(x[b.a as usize] - x[b.b as usize]));
                y[b.a as usize] += f;
                y[b.b as usize] -= f;
            }
            for i in 0..n {
                y[i] /= s.nodes[i].mass.max(1e-6);
            }
            let norm = y.iter().map(|v| v.length_squared()).sum::<f32>().sqrt();
            let xn = x.iter().map(|v| v.length_squared()).sum::<f32>().sqrt();
            lam = norm / xn;
            x = y.iter().map(|v| *v / norm).collect();
        }
        let (imax, _) = x.iter().enumerate().max_by(|a, b| a.1.length().total_cmp(&b.1.length())).unwrap();
        println!("omega_max*h = {:.3} (node {} {})", lam.sqrt() * 0.0005, imax, desc.nodes.get(imax).map_or("wheel", |n| n.name.as_str()));
        println!("  node {imax} mass {}", s.nodes[imax].mass);
        for b in s.beams.iter().filter(|b| !b.broken && (b.a as usize == imax || b.b as usize == imax)) {
            let o = if b.a as usize == imax { b.b } else { b.a } as usize;
            println!("  {:?} k {} c {} to node {o} m {} len {:.3}", b.kind, b.k, b.c, s.nodes[o].mass, s.nodes[imax].pos.distance(s.nodes[o].pos));
        }
        return;
    }
    if opt("trace=").is_some() {
        for pw in &desc.pressure_wheels {
            println!(
                "wheel {} values {:?} flags {:?}",
                pw.name, pw.values, pw.flags
            );
        }
        let n0 = desc.nodes.len();
        for step in 0..4000 {
            s.step();
            if s.last_contacts > 0 && step < 40 {
                println!("step {step}: {} self-collision contacts", s.last_contacts);
            }
            let bad = s
                .nodes
                .iter()
                .position(|n| !n.pos.is_finite() || n.vel.length() > opt("vmax=").map_or(30.0, |v| v.parse().unwrap()));
            if let Some(i) = bad {
                println!(
                    "step {step}: node {i} (wheel node {}) {:?} {}",
                    i >= n0,
                    s.nodes[i],
                    desc.nodes.get(i).map_or("", |n| n.name.as_str())
                );
                for (j, b) in s.beams.iter().enumerate() {
                    if b.a as usize == i || b.b as usize == i {
                        let o = if b.a as usize == i { b.b } else { b.a } as usize;
                        println!(
                            "  beam {j} other {} m {} {:?} ext {:?}",
                            desc.nodes.get(o).map_or("wheel", |n| n.name.as_str()),
                            s.nodes[o].mass,
                            b,
                            s.ext_of
                                .get(j)
                                .filter(|&&x| x != 0)
                                .map(|&x| s.beam_ext[x as usize - 1])
                        );
                    }
                }
                return;
            }
        }
        println!("no blow-up in 2 s");
        let mut ke: Vec<(f32, usize)> = s
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (0.5 * n.mass * n.vel.length_squared(), i))
            .collect();
        ke.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(e, i) in ke.iter().take(8) {
            println!(
                "  KE {e:.1} J node {i} {} m {} v {:?} y {:.3}",
                desc.nodes.get(i).map_or("wheel", |n| n.name.as_str()),
                s.nodes[i].mass,
                s.nodes[i].vel,
                s.nodes[i].pos.y
            );
        }
        return;
    }
    let t0 = std::time::Instant::now();
    report("rest 0 s", &s, 0.0);
    for k in 1..=(rest.ceil() as usize) {
        if opt("restwatch=").is_some() {
            for q in 0..10 {
                s.run(0.1);
                let (i, n) = s
                    .nodes
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.vel.length().total_cmp(&b.1.vel.length()))
                    .unwrap();
                let lat: f32 = s.nodes[desc.nodes.len()..].iter().map(|n| n.vel.x * n.mass).sum::<f32>()
                    / s.nodes[desc.nodes.len()..].iter().map(|n| n.mass).sum::<f32>();
                println!(
                    "  {:.1} s fastest node {} {} m {} v {:.2?} pos {:.3?}; wheel nodes' mean x velocity {:.4}",
                    (k - 1) as f32 + 0.1 * (q + 1) as f32,
                    i,
                    desc.nodes.get(i).map_or("wheel", |n| n.name.as_str()),
                    n.mass,
                    n.vel,
                    n.pos,
                    lat
                );
            }
            report(&format!("rest {k} s"), &s, 0.0);
            continue;
        }
        s.run(1.0);
        report(&format!("rest {k} s"), &s, 0.0);
    }
    println!("rest: {:.1} s wall", t0.elapsed().as_secs_f32());
    if opt("brokenlist=").is_some() {
        let name = |i: u32| desc.nodes.get(i as usize).map_or("wheel".to_string(), |n| n.name.clone());
        for b in s.beams.iter().filter(|b| b.broken) {
            println!("  broken {:?} {}-{} k {} strength {} group {}", b.kind, name(b.a), name(b.b), b.k, b.strength, b.group);
        }
    }
    let v = Vec3::new(0.0, 0.0, kmh / 3.6);
    let (mut s, _) = spawn(v);
    if opt("coast=").is_some() {
        // Coasting for 1 s: the mean deceleration's force (air drag, rolling).
        let vz = |s: &ExplicitBeams| s.nodes.iter().map(|n| n.mass * n.vel.z).sum::<f32>();
        let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
        {
            let mut l: Vec<(f32, usize)> = s.nodes.iter().enumerate().filter(|(i, n)| *i < desc.nodes.len() && n.flags & engine_softbody::explicit_beams::NODE_NO_GROUND == 0).map(|(i, n)| (n.pos.y, i)).collect();
            l.sort_by(|a, b| a.0.total_cmp(&b.0));
            let tyre_low = s.nodes[desc.nodes.len()..].iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            println!("  tyres' lowest {tyre_low:.3}; lowest collidable body nodes {:?}", l.iter().take(6).map(|(y, i)| format!("{} {y:.3}", desc.nodes[*i].name)).collect::<Vec<_>>());
        }
        for (k, t) in s.tyres.iter().enumerate() {
            let inside = |i: u32| i >= t.first && i < t.first + t.count;
            let mut partners: Vec<String> = s
                .beams
                .iter()
                .filter(|b| inside(b.a) != inside(b.b))
                .map(|b| {
                    let o = if inside(b.a) { b.b } else { b.a };
                    format!("{}:{:?}", desc.nodes.get(o as usize).map_or("wheel".to_string(), |n| n.name.clone()), b.kind)
                })
                .collect();
            partners.sort();
            partners.dedup();
            println!("  tyre {k} axle {:?}: outside partners {partners:?}", t.axle.map(|i| desc.nodes.get(i as usize).map_or("?", |n| n.name.as_str())));
        }
        if opt("coasttrace=").is_some() {
            let n0 = desc.nodes.len();
            for k in 0..30 {
                s.run(0.01);
                let body: f32 = s.nodes[..n0].iter().map(|n| n.mass * n.vel.z).sum::<f32>() / s.nodes[..n0].iter().map(|n| n.mass).sum::<f32>();
                let wheels: f32 = s.nodes[n0..].iter().map(|n| n.mass * n.vel.z).sum::<f32>() / s.nodes[n0..].iter().map(|n| n.mass).sum::<f32>().max(1e-6);
                let (i, fast) = s.nodes.iter().enumerate().max_by(|a, b| a.1.vel.length().total_cmp(&b.1.vel.length())).unwrap();
                println!("  {:.2} s: body vz {body:.2}, wheel nodes vz {wheels:.2}; fastest {} {:.1?}", 0.01 * (k + 1) as f32, desc.nodes.get(i).map_or("wheel", |n| n.name.as_str()), fast.vel);
            }
            return;
        }
        let mut low: Vec<(f32, usize)> = s.nodes.iter().enumerate().filter(|(_, n)| n.flags & engine_softbody::explicit_beams::NODE_NO_GROUND == 0).map(|(i, n)| (n.pos.y, i)).collect();
        low.sort_by(|a, b| a.0.total_cmp(&b.0));
        let body_low: Vec<String> = low.iter().filter(|(_, i)| *i < desc.nodes.len()).take(6).map(|(y, i)| format!("{} {y:.3}", desc.nodes[*i].name)).collect();
        println!("  lowest collidable body nodes: {body_low:?}");
        for &(y, i) in low.iter().take(4) {
            println!("  low node {i} {} y {y:.3} m {}", desc.nodes.get(i).map_or("wheel", |n| n.name.as_str()), s.nodes[i].mass);
        }
        s.run(0.5);
        // Non-tyre nodes on the ground (they scrape).
        let touching: Vec<String> = s
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.flags & engine_softbody::explicit_beams::NODE_TYRE == 0 && n.flags & engine_softbody::explicit_beams::NODE_NO_GROUND == 0 && n.pos.y < 0.003)
            .map(|(i, n)| format!("{} {:.3} v {:.2}", desc.nodes.get(i).map_or("?", |d| d.name.as_str()), n.pos.y, n.vel.z))
            .collect();
        println!("  non-tyre nodes on the ground at 0.5 s: {touching:?}");
        for t in &s.tyres {
            let half = t.count / 2;
            let hub = s.nodes[t.first as usize..(t.first + half) as usize].iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            let tread = s.nodes[(t.first + half) as usize..(t.first + t.count) as usize].iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            println!("  wheel span {}: lowest of first half {hub:.3}, of second half {tread:.3}", t.first);
        }
        for (k, t) in s.tyres.iter().enumerate() {
            let [a1, a2] = t.axle.map(|i| s.nodes[i as usize]);
            let c = 0.5 * (a1.pos + a2.pos);
            let vc = 0.5 * (a1.vel + a2.vel);
            let ax = (a2.pos - a1.pos).normalize();
            let mut w = 0.0;
            for i in t.first..t.first + t.count {
                let n = s.nodes[i as usize];
                let r = n.pos - c;
                let r = r - ax * r.dot(ax);
                w += r.cross(n.vel - vc).dot(ax) / r.length_squared().max(1e-6);
            }
            w /= t.count as f32;
            println!("  tyre {k}: spin {w:.2} rad/s about its axle; centre speed {:.2} m/s, as rolling {:.2} rad/s (radius {:.3})", vc.z, vc.z / t.radius, t.radius);
        }
        let p0 = vz(&s);
        s.run(1.0);
        let f = (p0 - vz(&s)) / 1.0;
        let v1 = vz(&s) / m;
        println!(
            "coast at {:.1} m/s: decelerating force {f:.0} N; as 0.5*rho*CdA*v^2, CdA = {:.2} m^2",
            v1,
            f / (0.5 * engine_softbody::explicit_jbeam_ext::AIR_DENSITY * v1 * v1)
        );
        return;
    }
    let front = s
        .nodes
        .iter()
        .map(|n| n.pos.z)
        .fold(f32::NEG_INFINITY, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.5, front + 0.3 + 1.0),
        Vec3::new(5.0, 3.0, 1.0),
    ));
    let ke0: f32 = s
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum();
    let steps = (crash / 0.1).ceil() as usize;
    for k in 1..=steps {
        s.run(0.1);
        if opt("crashwatch=").is_some() {
            let (i, n) = s
                .nodes
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.vel.length().total_cmp(&b.1.vel.length()))
                .unwrap();
            println!(
                "  {:.1} s fastest node {} {} m {} v {:.2} m/s",
                0.1 * k as f32,
                i,
                desc.nodes.get(i).map_or("wheel", |n| n.name.as_str()),
                n.mass,
                n.vel.length()
            );
        }
        if k % 5 == 0 || k == steps {
            report(&format!("crash {:.1} s", 0.1 * k as f32), &s, ke0);
        }
    }
}
