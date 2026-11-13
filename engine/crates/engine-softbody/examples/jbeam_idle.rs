//! A JBeam vehicle standing still on the explicit solver (CPU reference, drag off):
//! spawns it 2 cm above the ground at rest, runs `secs`, and lists the beams that
//! broke on their own (not through their break group) with the force, extension and
//! limits at that step, then the totals.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example jbeam_idle -- <vehicles-dir> <name> <config.pc> [secs=3] [path=native|truck] [objective=1]
//! ```
use engine_softbody::explicit_beams::{beam_force, ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (dir, name, pc) = (&a[0], &a[1], &a[2]);
    let opt = |k: &str| a.iter().find_map(|x| x.strip_prefix(k)).map(str::to_string);
    let secs: f32 = opt("secs=").map_or(3.0, |s| s.parse().unwrap());
    let path = opt("path=").unwrap_or_else(|| "native".into());
    let dir = std::path::Path::new(dir);
    let texts = engine_softbody::jbeam::read_vehicle_dir(dir, name).unwrap();
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        fast_inv_sqrt: opt("fast=").as_deref() != Some("0"),
        // `objective=1`: the centred damping's objective form (spec §39.6).
        centred_objective: opt("objective=").as_deref() == Some("1"),
        ..Default::default()
    };
    let desc = engine_softbody::jbeam::load_jbeam(&files, Some(&pcs)).unwrap();
    for w in &desc.wheels {
        println!(
            "wheel axle {}-{} arm {:?} radius {:.3} width {:.3}",
            desc.nodes[w.node1 as usize].name,
            desc.nodes[w.node2 as usize].name,
            w.arm.map(|a| desc.nodes[a as usize].name.clone()),
            w.radius,
            w.width
        );
    }
    let speed: f32 = opt("speed=").map_or(0.0, |s| s.parse().unwrap());
    let node_name = |i: u32| {
        desc.nodes
            .get(i as usize)
            .map_or(format!("#{i}"), |n| n.name.clone())
    };
    let mut sim = match path.as_str() {
        "truck" => {
            ExplicitBeams::from_jbeam(&files, Some(&pcs), Vec3::ZERO, Vec3::ZERO, settings)
                .unwrap()
                .0
        }
        _ => {
            let mut d = desc.clone();
            if opt("nohydro=").is_some() {
                d.beams
                    .retain(|b| b.kind != engine_softbody::desc::BeamKind::Hydro);
            }
            let tyres = engine_softbody::truck_export::TyreDefaults::for_vehicle(&d);
            let (s, w) =
                ExplicitBeams::from_desc_jbeam(&d, tyres, Vec3::ZERO, Vec3::ZERO, settings)
                    .unwrap();
            for x in &w {
                println!("warning: {x}");
            }
            s
        }
    };
    if opt("nofast=").is_some() {
        for b in &mut sim.beams {
            if b.kind == engine_softbody::explicit_beams::ExplicitKind::JbeamBounded {
                b.tc = f32::INFINITY;
            }
        }
    }
    if opt("selfcoll=").as_deref() == Some("0") {
        sim.candidates.clear();
    }
    println!("self-collision candidates: {}", sim.candidates.len());
    if opt("contacts=").is_some() {
        use engine_softbody::explicit_beams::TriFrame;
        let (mut same, mut other) = (0, 0);
        for t in &sim.triangles {
            let p = t.nodes.map(|i| sim.nodes[i as usize].pos);
            let f = TriFrame::new(p[0], p[1], p[2]);
            for &c in &sim.candidates {
                if t.nodes.contains(&c) {
                    continue;
                }
                if f.hit(sim.nodes[c as usize].pos, sim.collision_range)
                    .is_some()
                {
                    let g = desc.nodes[c as usize].self_group;
                    let shared = g != 0
                        && t.nodes
                            .iter()
                            .all(|&k| desc.nodes[k as usize].self_group == g);
                    if shared {
                        same += 1
                    } else {
                        other += 1
                    }
                    if !shared && other <= 15 {
                        let gn = |k: u32| {
                            desc.node_groups
                                .get((desc.nodes[k as usize].self_group as usize).wrapping_sub(1))
                                .cloned()
                                .unwrap_or_default()
                        };
                        println!(
                            "  spawn contact {} ({}) vs {}/{}/{} ({}/{}/{})",
                            node_name(c),
                            gn(c),
                            node_name(t.nodes[0]),
                            node_name(t.nodes[1]),
                            node_name(t.nodes[2]),
                            gn(t.nodes[0]),
                            gn(t.nodes[1]),
                            gn(t.nodes[2])
                        );
                    }
                }
            }
        }
        println!("spawn contacts: {same} with a triangle of the node's own group, {other} other");
    }
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    if speed > 0.0 {
        // Into a rigid wall 5 cm ahead, wider and taller than the vehicle.
        let v = Vec3::new(0.0, 0.0, speed);
        for n in &mut sim.nodes {
            n.vel = v;
        }
        sim.roll_tyres(v);
        let front = sim.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        sim.boxes
            .push(engine_softbody::explicit_beams::StaticBox::centered(
                Vec3::new(0.0, 1.5, front + 0.05 + 1.0),
                Vec3::new(4.0, 3.0, 1.0),
            ));
        println!("into a wall at {speed} m/s ({:.0} km/h)", speed * 3.6);
    }
    let steps = (secs * 2000.0) as u64;
    let every: u64 = opt("every=").map_or(2000, |s| s.parse().unwrap());
    let mut peak = 0.0f32;
    let mut peak_at = (0u64, 0u32);
    let mut own = 0;
    let mut ke_node = vec![0.0f64; sim.nodes.len()];
    for s in 0..steps {
        let before = sim.beams.clone();
        sim.step();
        for (j, b) in sim.beams.iter().enumerate() {
            if b.broken && !before[j].broken {
                let mut pre = before[j];
                let out = beam_force(
                    &mut pre,
                    &sim.nodes[b.a as usize],
                    &sim.nodes[b.b as usize],
                    true,
                );
                if !(out.check_break || pre.broken) && opt("all=").is_some() {
                    println!(
                        "step {s:6} beam {j:5} {}-{} {:?} broke with its group {} (F={:.0}, S={:.0})",
                        node_name(b.a), node_name(b.b), b.kind, b.group, out.force, before[j].strength
                    );
                }
                if out.check_break || pre.broken {
                    own += 1;
                    if own <= 40 {
                        println!(
                            "step {s:6} beam {j:5} {}-{} {:?} k={:.0} c={:.0} S={:.0} F={:.0} e={:.4} L={:.4} L0={:.4} group={}",
                            node_name(b.a), node_name(b.b), b.kind, b.k, b.c, before[j].strength, out.force, out.e,
                            before[j].rest, before[j].rest0, b.group
                        );
                    }
                }
            }
        }
        if let Some(w) = opt("trace=") {
            let from: u64 =
                opt("trace_from=").map_or(steps.saturating_sub(40), |x| x.parse().unwrap());
            let idx = desc
                .nodes
                .iter()
                .position(|n| n.name == w)
                .or_else(|| w.strip_prefix('#').and_then(|x| x.parse().ok()));
            if let (true, Some(i)) = (s >= from && s < from + 40, idx) {
                let n = &sim.nodes[i];
                let live: Vec<String> = sim.incidence[i]
                    .iter()
                    .filter(|&&j| !sim.beams[j as usize].broken)
                    .map(|&j| {
                        let b = &sim.beams[j as usize];
                        format!("{j}:{:?}:F={:.0}", b.kind, b.force)
                    })
                    .collect();
                println!(
                    "trace {w} step {s}: pos {:.4} {:.4} {:.4} vel {:.2} {:.2} {:.2} live {live:?}",
                    n.pos.x, n.pos.y, n.pos.z, n.vel.x, n.vel.y, n.vel.z
                );
            }
        }
        if s + 2000 >= steps {
            for (i, n) in sim.nodes.iter().enumerate() {
                ke_node[i] += (0.5 * n.mass * n.vel.length_squared()) as f64 / 2000.0;
            }
        }
        if (s + 1) % every == 0 {
            let ke: f32 = sim
                .nodes
                .iter()
                .map(|n| 0.5 * n.mass * n.vel.length_squared())
                .sum();
            let (vmax, at) = sim
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (n.vel.length(), i))
                .fold((0.0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a });
            let (broken, bent) = sim.damage();
            println!(
                "t={:5.1} s KE {ke:9.4} J, fastest node {} at {vmax:.4} m/s, broken {broken}, bent {bent}",
                (s + 1) as f32 / 2000.0,
                node_name(at as u32)
            );
        }
        for (i, n) in sim.nodes.iter().enumerate() {
            let v = n.vel.length();
            if v > peak {
                peak = v;
                peak_at = (s, i as u32);
            }
        }
        if sim.nodes.iter().any(|n| !n.pos.is_finite()) {
            println!("NaN at step {s}");
            break;
        }
    }
    if let Some(w) = opt("trace=") {
        use engine_softbody::explicit_beams::TriFrame;
        if let Some(c) = desc.nodes.iter().position(|n| n.name == w) {
            for t in &sim.triangles {
                if t.nodes.contains(&(c as u32)) {
                    continue;
                }
                let p = t.nodes.map(|i| sim.nodes[i as usize].pos);
                if let Some(h) =
                    TriFrame::new(p[0], p[1], p[2]).hit(sim.nodes[c].pos, 2.0 * sim.collision_range)
                {
                    let gn = |k: u32| {
                        desc.node_groups
                            .get((desc.nodes[k as usize].self_group as usize).wrapping_sub(1))
                            .cloned()
                            .unwrap_or_default()
                    };
                    println!(
                        "  {w} ({}) near {}/{}/{} ({}/{}/{}) depth {:.4}",
                        gn(c as u32),
                        node_name(t.nodes[0]),
                        node_name(t.nodes[1]),
                        node_name(t.nodes[2]),
                        gn(t.nodes[0]),
                        gn(t.nodes[1]),
                        gn(t.nodes[2]),
                        h.3
                    );
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..ke_node.len()).collect();
    order.sort_by(|&x, &y| ke_node[y].total_cmp(&ke_node[x]));
    let total: f64 = ke_node.iter().sum();
    println!("last second's mean KE {total:.3} J; top nodes:");
    for &i in order.iter().take(12) {
        println!(
            "  {:>12} {:.4} J (mass {:.2})",
            node_name(i as u32),
            ke_node[i],
            sim.nodes[i].mass
        );
    }
    {
        // Explicit stability: per node dt²·Σk/m over its live beams (single-DOF limit 4).
        let mut r: Vec<(f32, usize)> = (0..sim.nodes.len())
            .map(|i| {
                let ks: f32 = sim.incidence[i]
                    .iter()
                    .map(|&j| {
                        sim.beams[j as usize].k.max(
                            sim.beams[j as usize].limit_k.min(1e9)
                                * (sim.beams[j as usize].kind
                                    == engine_softbody::explicit_beams::ExplicitKind::JbeamBounded)
                                    as u8 as f32,
                        )
                    })
                    .sum();
                (0.0005f32 * 0.0005 * ks / sim.nodes[i].mass, i)
            })
            .collect();
        r.sort_by(|a, b| b.0.total_cmp(&a.0));
        print!("stiffest nodes (dt²Σk/m):");
        for &(x, i) in r.iter().take(8) {
            print!(" {}={x:.2}", node_name(i as u32));
        }
        println!();
    }
    for who in opt("show=")
        .unwrap_or_default()
        .split(',')
        .filter(|w| !w.is_empty())
    {
        let Some(i) = desc.nodes.iter().position(|n| n.name == who) else {
            continue;
        };
        let n = &sim.nodes[i];
        println!(
            "node {who}: mass {:.2} flags {} vel {:.3} height {:.4} (lowest {:.4})",
            n.mass,
            n.flags,
            n.vel.length(),
            n.pos.y,
            sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min)
        );
        for &j in &sim.incidence[i] {
            let b = &sim.beams[j as usize];
            let o = if b.a as usize == i { b.b } else { b.a };
            println!(
                "  beam {j} to {} (m {:.2}) {:?} k={:.0} c={:.0} L={:.4} L0={:.4} len={:.4} force {:.1} broken {}",
                node_name(o), sim.nodes[o as usize].mass, b.kind, b.k, b.c, b.rest, b.rest0, sim.nodes[b.a as usize].pos.distance(sim.nodes[b.b as usize].pos), b.force, b.broken
            );
        }
        for (k, e) in desc.elements.iter().enumerate() {
            if e.nodes.contains(&(i as u32)) {
                println!("  element {k} {:?}", e.kind);
            }
        }
        for s2 in &desc.slidenodes {
            if s2.node as usize == i {
                println!("  slide node on rail {}", s2.rail);
            }
        }
    }
    for (j, b) in sim
        .beams
        .iter()
        .enumerate()
        .filter(|(_, b)| b.bent())
        .take(12)
    {
        println!(
            "  bent {j} {}-{} {:?} k={:.0} rest {:.4} rest0 {:.4} deform {:.0}",
            node_name(b.a),
            node_name(b.b),
            b.kind,
            b.k,
            b.rest,
            b.rest0,
            b.tc
        );
    }
    println!(
        "peak node speed {peak:.2} m/s: node {} at step {} ({} live beams now)",
        node_name(peak_at.1),
        peak_at.0,
        sim.incidence[peak_at.1 as usize]
            .iter()
            .filter(|&&j| !sim.beams[j as usize].broken)
            .count()
    );
    let (broken, bent) = sim.damage();
    let ke: f32 = sim
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum();
    println!(
        "{name} {pc} ({path}): {} nodes, {} beams; after {secs} s{}: {broken} broken ({own} on their own), {bent} bent, KE {ke:.3} J, peak node speed {peak:.2} m/s, finite {}",
        sim.nodes.len(),
        sim.beams.len(),
        if speed > 0.0 { format!(" into a wall at {speed} m/s") } else { " at rest".into() },
        sim.nodes.iter().all(|n| n.pos.is_finite())
    );
}
