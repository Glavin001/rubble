//! Diagnostic: the visual skin's binding at rest. Per piece, the bind height
//! |h| distribution of vertices that lie on the collision skin (within 2 cm),
//! and the worst offenders with their bound part and nearest skin part.
//! `cargo run --release -p engine-softbody --example visual_bind_probe -- sedan1`
use engine_softbody::authoring::skin::showcase_visual;
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_skin::closest_on_triangle;
use glam::Vec3;
use std::collections::BTreeMap;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sedan1".into());
    if name == "all" {
        // Every car at every resolution: does the visual bind?
        for car in ["sedan", "sedan1", "hatchback"] {
            for res in [Resolution::Low, Resolution::Medium, Resolution::High] {
                let b = build(
                    &style(car).unwrap(),
                    &BuildParams {
                        resolution: res,
                        ..Default::default()
                    },
                );
                let r = b
                    .map_err(|e| format!("build: {e}"))
                    .and_then(|b| showcase_visual(&b).map(|v| v.positions.len()));
                println!("{car} {res:?}: {r:?}");
            }
        }
        return;
    }
    let b = build(
        &style(&name).unwrap(),
        &BuildParams {
            resolution: std::env::var("RES")
                .ok()
                .and_then(|r| Resolution::parse(&r))
                .unwrap_or(Resolution::Medium),
            ..Default::default()
        },
    )
    .unwrap();
    if let Ok(at) = std::env::var("NEAR") {
        // Skeleton nodes near a point in definition coordinates.
        let c: Vec<f32> = at.split(',').map(|x| x.parse().unwrap()).collect();
        let p = Vec3::new(c[0], c[1], c[2]);
        let sk = &b.skeleton;
        let on: std::collections::HashSet<u32> = sk.skin.iter().flatten().copied().collect();
        println!("shift {:?}", b.shift);
        let mut l: Vec<(f32, usize)> = sk
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.p.distance(p), i))
            .collect();
        l.sort_by(|a, c| a.0.total_cmp(&c.0));
        for &(d, i) in l.iter().take(20) {
            println!(
                "  n{i} {} at {:.3?} d {d:.3} skin {}",
                sk.parts[sk.nodes[i].part as usize],
                sk.nodes[i].p,
                on.contains(&(i as u32))
            );
        }
        return;
    }
    if let Ok(pn) = std::env::var("PART") {
        let sk = &b.skeleton;
        let on: std::collections::HashSet<u32> = sk.skin.iter().flatten().copied().collect();
        for (i, n) in sk.nodes.iter().enumerate() {
            if sk.parts[n.part as usize] == pn {
                println!("  n{i} at {:.3?} skin {}", n.p, on.contains(&(i as u32)));
            }
        }
        return;
    }
    if std::env::var("GROUPS").is_ok() {
        // Each structure group's node count and bounds (definition coordinates).
        let mut names: Vec<&String> = b.groups.keys().collect();
        names.sort();
        for n in names {
            let g = &b.groups[n];
            let (mut lo, mut hi) = (Vec3::splat(9.0), Vec3::splat(-9.0));
            for &i in g {
                lo = lo.min(b.skeleton.nodes[i as usize].p);
                hi = hi.max(b.skeleton.nodes[i as usize].p);
            }
            println!(
                "{n:>20} {:4} nodes {:.2?}..{:.2?} tris {}",
                g.len(),
                lo,
                hi,
                b.bind_tris.get(n).map_or(0, |t| t.len())
            );
        }
        return;
    }
    let v = showcase_visual(&b).unwrap();
    let sk = &b.skeleton;
    let node: Vec<Vec3> = v.node_rest.iter().map(|q| Vec3::from_array(*q)).collect();
    let near = |p: Vec3, tris: &[[u32; 3]]| -> (f32, usize) {
        let mut best = (f32::MAX, 0);
        for (ti, t) in tris.iter().enumerate() {
            let (q, _, _) = closest_on_triangle(
                p,
                node[t[0] as usize],
                node[t[1] as usize],
                node[t[2] as usize],
            );
            let d = p.distance(q);
            if d < best.0 {
                best = (d, ti);
            }
        }
        best
    };
    let part_of = |i: u32| {
        sk.nodes
            .get(i as usize)
            .map(|n| sk.parts[n.part as usize].clone())
            .unwrap_or("tyre".into())
    };
    if let Ok(pn) = std::env::var("PAIR") {
        // Inner stamping vertices whose frame part differs from the outer
        // vertex nearest them: where, and the skin distance at the outer point.
        let pof = |i: u32| {
            sk.nodes
                .get(i as usize)
                .map_or("tyre".to_string(), |n| sk.parts[n.part as usize].clone())
        };
        let maj = |t: [u32; 3]| {
            let p = t.map(pof);
            if p[1] == p[2] {
                p[1].clone()
            } else {
                p[0].clone()
            }
        };
        let outer: Vec<usize> = v
            .pieces
            .iter()
            .filter(|p| p.0.trim_end_matches('+') == pn)
            .flat_map(|p| (p.1 as usize)..(p.1 + p.2) as usize)
            .collect();
        let inner = v
            .pieces
            .iter()
            .find(|p| p.0 == format!("{pn}_inner"))
            .unwrap();
        let mut shown = 0;
        for k in (inner.1..inner.1 + inner.2).step_by(37) {
            let k = k as usize;
            let p = Vec3::from_array(v.positions[k]);
            let j = *outer
                .iter()
                .min_by(|a, b| {
                    Vec3::from_array(v.positions[**a])
                        .distance(p)
                        .total_cmp(&Vec3::from_array(v.positions[**b]).distance(p))
                })
                .unwrap();
            let (a, c) = (maj(v.bind_nodes[k]), maj(v.bind_nodes[j]));
            if a != c && shown < 15 {
                shown += 1;
                let q = Vec3::from_array(v.positions[j]);
                println!(
                    "inner {:.3?} on {a} {:?} h {:.3}; outer {:.3} m away on {c} {:?} h {:.3}",
                    p - b.shift,
                    v.bind_nodes[k].map(pof),
                    v.bind_coords[k][2],
                    q.distance(p),
                    v.bind_nodes[j].map(pof),
                    v.bind_coords[j][2]
                );
            }
        }
        return;
    }
    if let Ok(pn) = std::env::var("PIECE") {
        // One piece's vertices by frame part (majority), skin or node frame.
        let pof = |i: u32| {
            sk.nodes
                .get(i as usize)
                .map_or("tyre".to_string(), |n| sk.parts[n.part as usize].clone())
        };
        let skin: std::collections::HashSet<[u32; 3]> = sk
            .skin
            .iter()
            .map(|t| {
                let mut u = *t;
                u.sort_unstable();
                u
            })
            .collect();
        let mut h: BTreeMap<(String, bool), (usize, Vec3, Vec3)> = BTreeMap::new();
        for (name, first, count) in &v.pieces {
            if name.trim_end_matches('+') != pn {
                continue;
            }
            for i in *first..first + count {
                let t = v.bind_nodes[i as usize];
                let ps = t.map(pof);
                let maj = if ps[1] == ps[2] {
                    ps[1].clone()
                } else {
                    ps[0].clone()
                };
                let mut u = t;
                u.sort_unstable();
                let p = Vec3::from_array(v.positions[i as usize]) - b.shift;
                let e = h.entry((maj, skin.contains(&u))).or_insert((
                    0,
                    Vec3::splat(9.0),
                    Vec3::splat(-9.0),
                ));
                e.0 += 1;
                e.1 = e.1.min(p);
                e.2 = e.2.max(p);
            }
        }
        for ((p, sk_), (n, lo, hi)) in h {
            println!("{p:>12} skin-tri {sk_:5}: {n:6} {lo:.2?}..{hi:.2?}");
        }
        return;
    }
    if std::env::var("ROOT").is_ok() {
        // The old binding's reach: each exterior vertex (within 2 cm of the
        // collision skin) against its part's single-part skin triangles only
        // (the old per-part binder), and what lies under it instead.
        let pof = |i: u32| sk.nodes.get(i as usize).map(|n| n.part);
        let mut pure: std::collections::HashMap<u16, Vec<[u32; 3]>> = Default::default();
        for t in &sk.skin {
            if let (Some(a), Some(b), Some(c)) = (pof(t[0]), pof(t[1]), pof(t[2])) {
                if a == b && b == c {
                    pure.entry(a).or_default().push(*t);
                }
            }
        }
        let (mut n, mut far, mut mixed, mut other, mut nopure) = (0, 0, 0, 0, 0);
        let mut worst = 0.0f32;
        for (_, first, count) in &v.pieces {
            for i in (*first..first + count).step_by(5) {
                let i = i as usize;
                let p = Vec3::from_array(v.positions[i]) - b.shift;
                let bn = v.bind_nodes[i];
                let Some(pt) = pof(bn[0]).filter(|_| v.rigid[i] == 0) else {
                    continue;
                };
                let ps = bn.map(pof);
                let pt = if ps[1] == ps[2] {
                    ps[1].unwrap_or(pt)
                } else {
                    pt
                };
                if near(p, &sk.collision).0 > 0.02 {
                    continue;
                }
                n += 1;
                let Some(own) = pure.get(&pt) else {
                    nopure += 1;
                    continue;
                };
                let (d_old, _) = near(p, own);
                if d_old <= 0.1 {
                    continue;
                }
                far += 1;
                worst = worst.max(d_old);
                let (_, ts) = near(p, &sk.skin);
                let tp = sk.skin[ts].map(pof);
                if tp.iter().any(|&q| q != tp[0]) {
                    mixed += 1;
                } else if tp[0] != Some(pt) {
                    other += 1;
                }
            }
        }
        println!("exterior samples {n}: old binder > 10 cm {far} (worst {:.0} mm): under a shut-line (mixed) triangle {mixed}, under another part's {other}; part with no single-part triangle {nopure}", worst * 1e3);
        return;
    }
    if std::env::var("MISMATCH").is_ok() {
        // Exterior vertices farther than 3 cm from their frame: by piece,
        // bound part and the part whose skin is under them; count, max, box.
        let mut g: BTreeMap<(String, String, String), (usize, f32, Vec3, Vec3)> = BTreeMap::new();
        let mut total = 0;
        for (pname, first, count) in &v.pieces {
            for i in *first..first + count {
                let i = i as usize;
                if v.rigid[i] == 1 {
                    continue;
                }
                let p = Vec3::from_array(v.positions[i]);
                if near(p, &sk.collision).0 > 0.02 {
                    continue;
                }
                total += 1;
                let t = v.bind_nodes[i];
                let (q, _, _) = closest_on_triangle(
                    p,
                    node[t[0] as usize],
                    node[t[1] as usize],
                    node[t[2] as usize],
                );
                let d = p.distance(q);
                if d <= 0.03 {
                    continue;
                }
                let (_, ts) = near(p, &sk.skin);
                let st = sk.skin[ts].map(part_of);
                let bt = t.map(part_of);
                let maj = |a: [String; 3]| {
                    if a[1] == a[2] {
                        a[1].clone()
                    } else {
                        a[0].clone()
                    }
                };
                let e = g.entry((pname.clone(), maj(bt), maj(st))).or_insert((
                    0,
                    0.0,
                    Vec3::splat(9.0),
                    Vec3::splat(-9.0),
                ));
                let pd = p - b.shift;
                e.0 += 1;
                e.1 = e.1.max(d);
                e.2 = e.2.min(pd);
                e.3 = e.3.max(pd);
            }
        }
        println!("exterior vertices {total}");
        for ((pc, bp, sp), (n, m, lo, hi)) in g {
            println!(
                "{pc:>16} bound {bp:>10} skin {sp:>10}: {n:5} max {:4.0} mm box {:.2?}..{:.2?}",
                m * 1e3,
                lo,
                hi
            );
        }
        return;
    }
    if std::env::var("DIST").is_ok() {
        // Per piece: bind distance (vertex to its frame triangle) percentiles.
        for (pname, first, count) in &v.pieces {
            let ext = std::env::var("EXT").is_ok();
            let mut d: Vec<(f32, usize)> = (*first..first + count)
                .filter(|&i| v.rigid[i as usize] == 0)
                .filter(|&i| {
                    !ext || near(Vec3::from_array(v.positions[i as usize]), &sk.collision).0 <= 0.02
                })
                .map(|i| {
                    let i = i as usize;
                    let t = v.bind_nodes[i];
                    let p = Vec3::from_array(v.positions[i]);
                    let (q, _, _) = closest_on_triangle(
                        p,
                        node[t[0] as usize],
                        node[t[1] as usize],
                        node[t[2] as usize],
                    );
                    (p.distance(q), i)
                })
                .collect();
            if d.is_empty() {
                continue;
            }
            d.sort_by(|a, c| a.0.total_cmp(&c.0));
            let q = |f: f32| d[((d.len() - 1) as f32 * f) as usize].0 * 1e3;
            let w = d.last().unwrap();
            let p = Vec3::from_array(v.positions[w.1]) - b.shift;
            println!(
                "{pname:>20} n {:6} p50 {:5.1} p90 {:5.1} p99 {:5.1} max {:5.1} mm at {:.3?} parts {:?}",
                d.len(), q(0.5), q(0.9), q(0.99), q(1.0), p, v.bind_nodes[w.1].map(part_of)
            );
        }
        return;
    }
    let mut per: BTreeMap<String, (usize, usize, f32)> = BTreeMap::new();
    let mut worst: Vec<(f32, usize, String)> = Vec::new();
    for (pname, first, count) in &v.pieces {
        for i in (*first..first + count).step_by(7) {
            let i = i as usize;
            if v.rigid[i] == 1 {
                continue;
            }
            let p = Vec3::from_array(v.positions[i]);
            let (dc, _) = near(p, &sk.collision);
            if dc > 0.02 {
                continue;
            }
            let h = v.bind_coords[i][2].abs();
            let e = per.entry(pname.clone()).or_default();
            e.0 += 1;
            if h > 0.02 {
                e.1 += 1;
            }
            e.2 = e.2.max(h);
            if h > 0.05 {
                worst.push((h, i, pname.clone()));
            }
        }
    }
    println!("piece: on-collision samples, |h|>2cm, max |h| (m)");
    for (k, (n, f, m)) in &per {
        if *f > 0 {
            println!("  {k}: {n}, {f}, {m:.3}");
        }
    }
    worst.sort_by(|a, c| c.0.total_cmp(&a.0));
    for (h, i, pname) in worst.iter().take(25) {
        let p = Vec3::from_array(v.positions[*i]);
        let (ds, ts) = near(p, &sk.skin);
        let t = sk.skin[ts];
        let bn = v.bind_nodes[*i];
        println!(
            "  {pname} v{i} at {:.3?} h {h:.3} bound {:?} parts {:?}; nearest skin tri {ds:.3} m parts {:?}; coords {:?}",
            p,
            bn,
            bn.map(part_of),
            t.map(part_of),
            v.bind_coords[*i]
        );
    }
    if let Ok(at) = std::env::var("AT") {
        let c: Vec<f32> = at.split(',').map(|x| x.parse().unwrap()).collect();
        let p = Vec3::new(c[0], c[1], c[2]);
        println!("nodes within 0.2 m of {p:?}:");
        for (i, q) in node.iter().enumerate() {
            if q.distance(p) < 0.2 {
                println!(
                    "  n{i} {} at {:.3?} d {:.3}",
                    part_of(i as u32),
                    q,
                    q.distance(p)
                );
            }
        }
        println!("skin tris within 0.05 m:");
        for t in &sk.skin {
            let (q, _, _) = closest_on_triangle(
                p,
                node[t[0] as usize],
                node[t[1] as usize],
                node[t[2] as usize],
            );
            if q.distance(p) < 0.05 {
                println!("  {:?} {:?} d {:.3}", t, t.map(part_of), q.distance(p));
            }
        }
        let tot: usize = per.values().map(|e| e.0).sum();
        let bad: usize = per.values().map(|e| e.1).sum();
        println!("on-collision samples {tot}, |h| > 2 cm {bad}");
    }
}

#[allow(dead_code)]
fn unused() {}
