//! vehicle_c2c_overlap's spawn diagnostics (car A, before the run).

use super::*;

/// NEAR=name (or *): the collision triangles within 6 cm of a node (self-collision
/// suspects at rest).
pub(super) fn near_report(d: &SoftVehicleDesc, sa: &ExplicitBeams) -> Result<(), String> {
    if let Ok(name) = std::env::var("NEAR") {
    // The collision triangles within 6 cm of a node (self-collision
    // suspects at rest).
    let s = &sa;
    let all: Vec<usize> = if name == "*" {
        (0..s.nodes.len()).collect()
    } else {
        vec![d
            .nodes
            .iter()
            .position(|n| n.name == name)
            .ok_or("no such node")?]
    };
    let tri_has = std::env::var("NEAR_TRI").unwrap_or_default();
    for i in all {
        let name = d.nodes[i].name.clone();
        let p = s.nodes[i].pos;
        for t in 0..s.triangles.len() {
            let n = s.triangles[t].nodes;
            if n.contains(&(i as u32))
                || !n
                    .iter()
                    .any(|&k| d.nodes[k as usize].name.starts_with(&tri_has))
            {
                continue;
            }
            let [a, b, c] = n.map(|k| s.nodes[k as usize].pos);
            let q = closest_on_tri(p, a, b, c);
            if q.distance(p) < 0.06 {
                println!(
                    "  {name} {:.3} m from triangle {t} {:?}",
                    q.distance(p),
                    n.map(|k| d.nodes[k as usize].name.clone())
                );
            }
        }
    }
}
    Ok(())
}

/// Spawn diagnostics of car A: the collision skin's edges, holes and openings.
pub(super) fn skin_report(design: &str, d: &SoftVehicleDesc, sa: &ExplicitBeams) {
    let s = sa;
    let live: Vec<usize> = (0..s.triangles.len())
        .filter(|&t| !s.tri_anti_only(t))
        .collect();
    let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
    for &t in &live {
        let n = s.triangles[t].nodes;
        for k in 0..3 {
            let (x, y) = (n[k], n[(k + 1) % 3]);
            *edges.entry((x.min(y), x.max(y))).or_default() += 1;
        }
    }
    let pos = |i: u32| s.origin + s.nodes[i as usize].pos;
    let open: Vec<(u32, u32)> = edges
        .iter()
        .filter(|(_, &c)| c == 1)
        .map(|(&e, _)| e)
        .collect();
    let nonman = edges.values().filter(|&&c| c > 2).count();
    let nose = (0..s.nodes.len() as u32)
        .map(|i| pos(i).z)
        .fold(f32::MIN, f32::max);
    let front = nose - 1.0;
    // Open edges along the car (0.5 m stations from the nose) and by height.
    let mut st: std::collections::BTreeMap<i32, (usize, f32)> = Default::default();
    for (x, y) in &open {
        let m = 0.5 * (pos(*x) + pos(*y));
        let e = st.entry(((nose - m.z) / 0.5).floor() as i32).or_default();
        e.0 += 1;
        e.1 += pos(*x).distance(pos(*y));
    }
    println!(
        "  open edges by 0.5 m station from the nose: {:?}",
        st.iter()
            .map(|(k, (n, l))| format!("{:.1}: {n} ({l:.1} m)", *k as f32 * 0.5))
            .collect::<Vec<_>>()
    );
    let mut hy: std::collections::BTreeMap<i32, usize> = Default::default();
    for (x, y) in &open {
        let m = 0.5 * (pos(*x) + pos(*y));
        *hy.entry((m.y / 0.2).floor() as i32).or_default() += 1;
    }
    println!(
        "  open edges by 0.2 m height band: {:?}",
        hy.iter()
            .map(|(k, n)| format!("{:.1}: {n}", *k as f32 * 0.2))
            .collect::<Vec<_>>()
    );
    let open_front: Vec<&(u32, u32)> = open
        .iter()
        .filter(|(x, y)| pos(*x).z.max(pos(*y).z) > front)
        .collect();
    {
        // Winding consistency: an edge two triangles share should be walked in
        // opposite directions; the signed volume's sign says which way the
        // normals ((p1 − p0) × (p2 − p0)) face (> 0 outward).
        let mut dir: HashMap<(u32, u32), Vec<bool>> = HashMap::new();
        let mut vol = 0.0f64;
        for &t in &live {
            let n = s.triangles[t].nodes;
            for k in 0..3 {
                let (x, y) = (n[k], n[(k + 1) % 3]);
                dir.entry((x.min(y), x.max(y))).or_default().push(x < y);
            }
            let (p0, p1, p2) = (pos(n[0]), pos(n[1]), pos(n[2]));
            vol += (p0.dot(p1.cross(p2)) / 6.0) as f64;
        }
        let same = dir.values().filter(|v| v.len() == 2 && v[0] == v[1]).count();
        let pairs = dir.values().filter(|v| v.len() == 2).count();
        // Per triangle: the normal at its centre against the outward test (the
        // winding just in front of and behind it).
        let h = hull(s);
        let mut inward = 0;
        for &t in &live {
            let n = s.triangles[t].nodes;
            let (p0, p1, p2) = (pos(n[0]), pos(n[1]), pos(n[2]));
            let c = (p0 + p1 + p2) / 3.0;
            let nn = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            if winding(c + nn * 0.01, &h) > winding(c - nn * 0.01, &h) + 0.2 {
                inward += 1;
            }
        }
        println!(
            "  winding: {same} of {pairs} shared edges walked the same way by both triangles; signed volume {vol:.3} m^3; triangles facing in (winding higher in front) {inward} of {}",
            live.len()
        );
    }
    if std::env::var("LOOPS").is_ok() {
        // LOOPS=1: the skin's holes as loops of open edges (chained end to end), each
        // with its length, centre, extent and the parts of its nodes.
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(x, y) in &open {
            adj.entry(x).or_default().push(y);
            adj.entry(y).or_default().push(x);
        }
        let mut used: std::collections::HashSet<(u32, u32)> = Default::default();
        let mut loops: Vec<Vec<u32>> = Vec::new();
        let mut starts: Vec<(u32, u32)> = open.clone();
        starts.sort();
        for &(x, y) in &starts {
            if used.contains(&(x.min(y), x.max(y))) {
                continue;
            }
            let mut lp = vec![x, y];
            used.insert((x.min(y), x.max(y)));
            let mut cur = y;
            loop {
                let next = adj[&cur].iter().copied().find(|&z| !used.contains(&(cur.min(z), cur.max(z))));
                let Some(z) = next else { break };
                used.insert((cur.min(z), cur.max(z)));
                if z == x {
                    break;
                }
                lp.push(z);
                cur = z;
            }
            loops.push(lp);
        }
        loops.sort_by(|a, b| {
            let len = |l: &Vec<u32>| (0..l.len()).map(|k| pos(l[k]).distance(pos(l[(k + 1) % l.len()]))).sum::<f32>();
            len(b).total_cmp(&len(a))
        });
        for lp in &loops {
            let len: f32 = (0..lp.len()).map(|k| pos(lp[k]).distance(pos(lp[(k + 1) % lp.len()]))).sum();
            let c = lp.iter().map(|&i| pos(i)).sum::<Vec3>() / lp.len() as f32;
            let lo = lp.iter().map(|&i| pos(i)).fold(Vec3::splat(f32::MAX), Vec3::min);
            let hi = lp.iter().map(|&i| pos(i)).fold(Vec3::splat(f32::MIN), Vec3::max);
            let mut parts: Vec<String> = lp
                .iter()
                .map(|&i| d.nodes[i as usize].name.trim_end_matches(|ch: char| ch.is_ascii_digit() || ch == '_').to_string())
                .collect();
            parts.sort();
            parts.dedup();
            println!(
                "  hole: {} edges, {:.2} m round, centre ({:.2}, {:.2}, {:.2} from the nose), extent {:.2} x {:.2} x {:.2}; parts {:?}",
                lp.len(),
                len,
                c.x,
                c.y,
                c.z - nose,
                hi.x - lo.x,
                hi.y - lo.y,
                hi.z - lo.z,
                parts
            );
        }
    }
    if std::env::var("OPEN").is_ok() {
        // OPEN=1: the front metre's open edges by node name (where the hull has holes).
        let mut v: Vec<String> = open_front
            .iter()
            .map(|(x, y)| {
                format!(
                    "{}-{} ({:.2} m, y {:.2})",
                    d.nodes[*x as usize].name,
                    d.nodes[*y as usize].name,
                    pos(*x).distance(pos(*y)),
                    0.5 * (pos(*x).y + pos(*y).y)
                )
            })
            .collect();
        v.sort();
        println!("  front metre's open edges: {v:?}");
    }
    let open_len: f32 = open.iter().map(|(x, y)| pos(*x).distance(pos(*y))).sum();
    let open_front_len: f32 = open_front
        .iter()
        .map(|(x, y)| pos(*x).distance(pos(*y)))
        .sum();
    let h = hull(s);
    let centre = {
        let (lo, hi) = aabb(&h);
        0.5 * (lo + hi)
    };
    println!(
        "{design}: {} nodes, {} contact points, {} collision triangles ({} anti-clip only); \
         edges {} ({} open, {:.1} m; {} non-manifold); front metre: {} open edges, {:.2} m; \
         winding at the box centre {:.2}",
        s.nodes.len(),
        s.contact_points.len(),
        live.len(),
        s.triangles.len() - live.len(),
        edges.len(),
        open.len(),
        open_len,
        nonman,
        open_front.len(),
        open_front_len,
        winding(centre, &h)
    );
}

/// Spawn diagnostics of car A: the render skin against the collision skin, the tyres,
/// the front's contact points.
pub(super) fn cover_report(d: &SoftVehicleDesc, sa: &ExplicitBeams, vis: &[ShowcaseVisual]) -> Result<(), String> {
        let s = sa;
        let pos = |i: u32| s.origin + s.nodes[i as usize].pos;
        let nose = (0..s.nodes.len() as u32).map(|i| pos(i).z).fold(f32::MIN, f32::max);
        let front = nose - 1.0;
        let h = hull(s);
        // The front's render skin (paint) against the collision skin.
        let now: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
        let (p, _) = deform(&vis[0], &now);
        let mut cov: Vec<f32> = p
            .iter()
            .zip(&vis[0].material)
            .filter(|(q, &m)| m == mat::PAINT && (s.origin + **q).z > front)
            .map(|(q, _)| dist(s.origin + *q, &h))
            .collect();
        if let Ok(dir) = std::env::var("DUMP") {
            // Every render vertex (material, position, distance to the skin) and
            // the collision triangles, for plots.
            let mut out = String::from("m,x,y,z,d\n");
            for (q, &m) in p.iter().zip(&vis[0].material) {
                let q = s.origin + *q;
                if q.z > front - 1.0 {
                    out += &format!("{m},{:.4},{:.4},{:.4},{:.4}\n", q.x, q.y, q.z, dist(q, &h));
                }
            }
            std::fs::write(format!("{dir}/render.csv"), out).map_err(|e| e.to_string())?;
            let mut out = String::from("ax,ay,az,bx,by,bz,cx,cy,cz\n");
            for t in &h {
                out += &format!(
                    "{},{},{},{},{},{},{},{},{}\n",
                    t[0].x, t[0].y, t[0].z, t[1].x, t[1].y, t[1].z, t[2].x, t[2].y, t[2].z
                );
            }
            std::fs::write(format!("{dir}/tris.csv"), out).map_err(|e| e.to_string())?;
            let mut out = String::from("name,x,y,z\n");
            for (i, n) in s.nodes.iter().enumerate() {
                let q = s.origin + n.pos;
                out += &format!("{},{},{},{}\n", d.nodes[i].name, q.x, q.y, q.z);
            }
            std::fs::write(format!("{dir}/nodes.csv"), out).map_err(|e| e.to_string())?;
        }
        {
            // Head-on rays (along −z, 2 cm grid over the frontal area): the
            // first render surface (anything but the cabin's interior) against
            // the first collision triangle. `gap` = how far behind the render
            // surface the collision skin is (or none within the car).
            let g = 0.02f32;
            let (nx, ny) = (110usize, 80usize);
            let cell = |x: f32, y: f32| -> Option<usize> {
                let (i, j) = (((x + 1.1) / g).floor(), (y / g).floor());
                (i >= 0.0 && j >= 0.0 && (i as usize) < nx && (j as usize) < ny)
                    .then(|| j as usize * nx + i as usize)
            };
            let raster = |tris: &mut dyn Iterator<Item = [Vec3; 3]>| -> Vec<f32> {
                let mut zb = vec![f32::MIN; nx * ny];
                for [a, b, c] in tris {
                    let lo = a.min(b).min(c);
                    let hi = a.max(b).max(c);
                    let den = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
                    if den.abs() < 1e-12 {
                        continue;
                    }
                    let mut x = ((lo.x + 1.1) / g).floor() * g - 1.1 + 0.5 * g;
                    while x <= hi.x {
                        let mut y = (lo.y / g).floor() * g + 0.5 * g;
                        while y <= hi.y {
                            let l1 = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / den;
                            let l2 = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / den;
                            let l3 = 1.0 - l1 - l2;
                            if l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0 {
                                if let Some(k) = cell(x, y) {
                                    zb[k] = zb[k].max(l1 * a.z + l2 * b.z + l3 * c.z);
                                }
                            }
                            y += g;
                        }
                        x += g;
                    }
                }
                zb
            };
            let interior = |m: u8| {
                matches!(
                    m,
                    mat::SEAT | mat::DASH | mat::DOORCARD | mat::HEADLINER | mat::CARPET
                )
            };
            let rp: Vec<Vec3> = p.iter().map(|q| s.origin + *q).collect();
            let zr = raster(
                &mut vis[0]
                    .indices
                    .chunks(3)
                    .filter(|t| !interior(vis[0].material[t[0] as usize]))
                    .map(|t| [rp[t[0] as usize], rp[t[1] as usize], rp[t[2] as usize]]),
            );
            let zc = raster(&mut h.iter().copied());
            let mut gaps: Vec<f32> = Vec::new();
            let (mut none, mut over) = (0usize, 0usize);
            let mut csv = String::from("x,y,zr,zc\n");
            for k in 0..nx * ny {
                if zr[k] == f32::MIN {
                    continue;
                }
                let gap = if zc[k] == f32::MIN {
                    f32::INFINITY
                } else {
                    zr[k] - zc[k]
                };
                if gap.is_infinite() {
                    none += 1;
                } else {
                    gaps.push(gap);
                    if gap > 0.15 {
                        over += 1;
                    }
                }
                let (x, y) = (
                    (k % nx) as f32 * g - 1.1 + 0.5 * g,
                    (k / nx) as f32 * g + 0.5 * g,
                );
                csv += &format!(
                    "{x:.3},{y:.3},{:.4},{:.4}\n",
                    zr[k] - nose,
                    if zc[k] == f32::MIN {
                        -9.0
                    } else {
                        zc[k] - nose
                    }
                );
            }
            let n = gaps.len() + none;
            println!(
                "  head-on rays ({n} cells of 2 cm hitting the car): collision skin behind the first render surface p50 {:.0} / p90 {:.0} / p99 {:.0} mm; > 15 cm {} cells ({:.1} %); no collision triangle at all {} ({:.1} %)",
                pct(&mut gaps, 0.5) * 1e3,
                pct(&mut gaps, 0.9) * 1e3,
                pct(&mut gaps, 0.99) * 1e3,
                over,
                100.0 * over as f32 / n as f32,
                none,
                100.0 * none as f32 / n as f32
            );
            if let Ok(dir) = std::env::var("DUMP") {
                std::fs::write(format!("{dir}/rays.csv"), csv).map_err(|e| e.to_string())?;
            }
        }
        let n_front = cov.len();
        let (p50, p90, p99, mx) = (
            pct(&mut cov, 0.5),
            pct(&mut cov, 0.9),
            pct(&mut cov, 0.99),
            pct(&mut cov, 1.0),
        );
        let outside = p
            .iter()
            .zip(&vis[0].material)
            .filter(|(q, &m)| m == mat::PAINT && (s.origin + **q).z > front)
            .filter(|(q, _)| winding(s.origin + **q, &h) < 0.5)
            .count();
        println!(
            "  front metre's paint vertices ({n_front}): distance to the collision skin p50 {:.0} / p90 {:.0} / p99 {:.0} / max {:.0} mm; outside the hull {} ({:.0} %)",
            p50 * 1e3,
            p90 * 1e3,
            p99 * 1e3,
            mx * 1e3,
            outside,
            100.0 * outside as f32 / n_front.max(1) as f32
        );
        Ok(())
}

/// Spawn diagnostics of car A: the tyres' render vertices against the hull, and the front's
/// contact points.
pub(super) fn tyre_report(d: &SoftVehicleDesc, sa: &ExplicitBeams, vis: &[ShowcaseVisual]) -> Result<(), String> {
        let s = sa;
        let pos = |i: u32| s.origin + s.nodes[i as usize].pos;
        let nose = (0..s.nodes.len() as u32).map(|i| pos(i).z).fold(f32::MIN, f32::max);
        let front = nose - 1.0;
        let h = hull(s);
        let now: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
        let (p, _) = deform(&vis[0], &now);
        // Tyres: their render vertices against the hull and the wheels.
        let tyre: Vec<Vec3> = p
            .iter()
            .zip(&vis[0].material)
            .filter(|(_, &m)| m == mat::TYRE)
            .map(|(q, _)| s.origin + *q)
            .collect();
        let tyre_in = tyre.iter().filter(|q| winding(**q, &h) >= 0.5).count();
        let tz = tyre.iter().map(|q| q.z).fold(f32::MIN, f32::max);
        println!(
            "  tyres: {} render vertices, {} inside the hull; front tyre's front at {:.3} m behind the nose; wheels {:?}",
            tyre.len(),
            tyre_in,
            nose - tz,
            d.wheels.iter().map(|w| (d.nodes[w.node1 as usize].name.clone(), w.radius, w.width)).collect::<Vec<_>>()
        );
        // Contact points at the front: which nodes.
        let mut fronts: Vec<(f32, String)> = s
            .contact_points
            .iter()
            .filter(|&&i| pos(i).z > front)
            .map(|&i| (pos(i).z, d.nodes[i as usize].name.clone()))
            .collect();
        fronts.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!(
            "  contact points in the front metre: {} (of {} nodes there)",
            fronts.len(),
            (0..s.nodes.len())
                .filter(|&i| pos(i as u32).z > front)
                .count()
        );
        let nc: Vec<String> = (0..s.nodes.len() as u32)
            .filter(|&i| pos(i).z > front && !s.contact_points.contains(&i))
            .map(|i| d.nodes[i as usize].name.clone())
            .collect();
        println!("  front metre nodes that are not contact points: {nc:?}");
        Ok(())
}
