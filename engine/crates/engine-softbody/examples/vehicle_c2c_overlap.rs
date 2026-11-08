//! How far two rigged cars get inside each other head on (CPU reference): the
//! `vehicle_c2c` / `vehicle_film c2c` scene (two cars of a design, full
//! overlap, `km/h` each, raycast wheels, vehicle contact, surface contact on),
//! measured every 4 ms. "Inside" is the generalised winding number (> 0.5)
//! of the other car's live collision triangles (its closed-or-not collision
//! hull); depth is the distance to the nearest of those triangles.
//!
//! Per direction (A in B, B in A):
//! * `nodes`: the car's contact points inside the other's hull (count, deepest);
//!   each node that enters is classified by its path since the last sample:
//!   through a live triangle ("face") or not ("gap": between triangles, a hole
//!   in the hull, or a dead triangle);
//! * `paint` / `tyre`: render vertices (paint, tyre) inside the other's hull
//!   (count, p99 and deepest);
//! * `edges`: the car's collision-skin edges through the other's triangles;
//! * `slab`: the render surfaces' overlap along the cars' axis (5 cm bins
//!   across the car: A's frontmost vertex beyond B's frontmost), the deepest bin
//!   and the overlapping volume.
//!
//! At spawn it prints the collision hull's open edges (edges on one triangle)
//! and how well the front's render skin is covered by collision triangles.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example vehicle_c2c_overlap -- [design] [km/h] [seconds]
//! ```
use engine_softbody::authoring::rig;
use engine_softbody::authoring::skin::{deform, mat, ShowcaseVisual};
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::vehicle_compare::settings;
use glam::Vec3;
use std::collections::HashMap;

struct Loaded {
    desc: SoftVehicleDesc,
    refs: Vec<(String, String)>,
    pc: Option<String>,
    built: rig::RiggedVehicle,
}

fn load(design: &str) -> Result<Loaded, String> {
    let built = rig::rig_vehicle(design, None, None)?;
    let files = rig::mod_files(&built);
    let refs: Vec<(String, String)> = files
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
        .collect();
    let pc = files
        .iter()
        .find(|(p, _)| p.ends_with(".pc"))
        .map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let r: Vec<(&str, &str)> = refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let desc = engine_softbody::jbeam::load_jbeam(&r, pc.as_deref())?;
    Ok(Loaded {
        desc,
        refs,
        pc,
        built,
    })
}

fn visual(
    l: &Loaded,
    desc: &SoftVehicleDesc,
    sim: &ExplicitBeams,
) -> Result<ShowcaseVisual, String> {
    let r: Vec<(&str, &str)> = l
        .refs
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let flex = engine_softbody::jbeam::assembled_rows(&r, l.pc.as_deref(), "flexbodies")?;
    let g = &l.built.input.glb;
    let meshes: Vec<_> = g
        .meshes
        .iter()
        .map(|m| engine_softbody::flexskin::glb_mesh(g, m))
        .collect();
    let paint = engine_softbody::glb::paint_of(g).unwrap_or([0.5, 0.05, 0.04]);
    let (v, _) = engine_softbody::flexmesh::jbeam_visual_skinned(
        &flex,
        &[],
        desc,
        sim,
        &meshes,
        &Default::default(),
        paint,
        Some(&l.built.skin),
    )?;
    Ok(v)
}

fn spawn(desc: &SoftVehicleDesc, at: Vec3, vel: Vec3) -> Result<ExplicitBeams, String> {
    let lift = 0.02
        - desc
            .nodes
            .iter()
            .map(|n| n.position[1])
            .fold(f32::MAX, f32::min);
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(desc, at + Vec3::Y * lift, vel, settings())?;
    s.surface_contact = std::env::var("SURFACE_CONTACT").map_or(true, |v| v != "0");
    s.finish();
    Ok(s)
}

/// A car's live collision triangles (not broken, not anti-clip only), world.
fn hull(s: &ExplicitBeams) -> Vec<[Vec3; 3]> {
    (0..s.triangles.len())
        .filter(|&t| !s.tri_dead.get(t).copied().unwrap_or(false) && !s.tri_anti_only(t))
        .map(|t| {
            s.triangles[t]
                .nodes
                .map(|i| s.origin + s.nodes[i as usize].pos)
        })
        .collect()
}

fn winding(p: Vec3, tris: &[[Vec3; 3]]) -> f32 {
    let mut w = 0.0f64;
    for t in tris {
        let (a, b, c) = (t[0] - p, t[1] - p, t[2] - p);
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        let num = a.dot(b.cross(c)) as f64;
        let den = (la * lb * lc + a.dot(b) * lc + a.dot(c) * lb + b.dot(c) * la) as f64;
        w += 2.0 * num.atan2(den);
    }
    (w / (4.0 * std::f64::consts::PI)) as f32
}

fn closest_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    // Ericson, Real-Time Collision Detection 5.1.5.
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let den = 1.0 / (va + vb + vc);
    a + ab * (vb * den) + ac * (vc * den)
}

fn dist(p: Vec3, tris: &[[Vec3; 3]]) -> f32 {
    tris.iter()
        .map(|t| closest_on_tri(p, t[0], t[1], t[2]).distance_squared(p))
        .fold(f32::MAX, f32::min)
        .sqrt()
}

/// Whether segment p→q crosses triangle t (Möller–Trumbore, both ends).
fn seg_tri(p: Vec3, q: Vec3, t: &[Vec3; 3]) -> bool {
    let d = q - p;
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let h = d.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-12 {
        return false;
    }
    let f = 1.0 / a;
    let s = p - t[0];
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let qv = s.cross(e1);
    let v = f * d.dot(qv);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let tt = f * e2.dot(qv);
    (0.0..=1.0).contains(&tt)
}

fn aabb(tris: &[[Vec3; 3]]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for t in tris {
        for p in t {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    (lo, hi)
}

fn inside_box(p: Vec3, (lo, hi): (Vec3, Vec3)) -> bool {
    p.cmpge(lo).all() && p.cmple(hi).all()
}

fn pct(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * q).round() as usize]
}

#[derive(Default, Clone, Copy)]
struct Dir {
    nodes: usize,
    node_depth: f32,
    face: usize,
    gap: usize,
    paint: usize,
    paint_p99: f32,
    paint_depth: f32,
    tyre: usize,
    tyre_depth: f32,
    edges: usize,
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let design = a.first().cloned().unwrap_or_else(|| "sedan".into());
    let kmh: f32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(50.0);
    let secs: f32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.6);
    let l = load(&design)?;
    let d = &l.desc;
    let zmax = d
        .nodes
        .iter()
        .map(|n| n.position[2])
        .fold(f32::MIN, f32::max);
    let v = kmh / 3.6;
    let mut tb = d.clone();
    for n in &mut tb.nodes {
        n.position = [-n.position[0], n.position[1], -n.position[2]];
    }
    let sa = spawn(d, Vec3::new(0.0, 0.0, -0.025 - zmax), Vec3::Z * v)?;
    let sb = spawn(&tb, Vec3::new(0.0, 0.0, 0.025 + zmax), -Vec3::Z * v)?;
    let vis: Vec<ShowcaseVisual> = [d, &tb]
        .iter()
        .map(|dd| {
            let (probe, _) =
                ExplicitBeams::from_desc_jbeam_raycast(dd, Vec3::ZERO, Vec3::ZERO, settings())?;
            visual(&l, dd, &probe)
        })
        .collect::<Result<_, String>>()?;

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
    // ---- Spawn diagnostics (car A) ----
    if std::env::var("MESHES").is_ok() {
        // The render meshes at the front (glTF: +Y up, +Z forward): each mesh's
        // vertex count and extent in the front 0.4 m.
        let m = &l.built.input.model;
        let zf = m
            .parts
            .iter()
            .flat_map(|p| p.pos.iter().map(|q| q.z))
            .fold(f32::MIN, f32::max);
        for p in &m.parts {
            let v: Vec<&Vec3> = p.pos.iter().filter(|q| q.z > zf - 0.4).collect();
            if v.is_empty() {
                continue;
            }
            let lo = v.iter().fold(Vec3::splat(f32::MAX), |a, q| a.min(**q));
            let hi = v.iter().fold(Vec3::splat(f32::MIN), |a, q| a.max(**q));
            println!(
                "  mesh {:<22} {:6} vertices in the front 0.4 m: x {:.2}..{:.2} y {:.2}..{:.2} z {:.2}..{:.2} (nose {:.2})",
                p.name, v.len(), lo.x, hi.x, lo.y, hi.y, lo.z - zf, hi.z - zf, zf
            );
        }
    }
    {
        let s = &sa;
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
    }

    let mut w = ExplicitBeamsWorld::new(vec![sa, sb]);
    let edges_of = |s: &ExplicitBeams| -> Vec<(u32, u32)> {
        let mut e: Vec<(u32, u32)> = (0..s.triangles.len())
            .filter(|&t| !s.tri_dead.get(t).copied().unwrap_or(false) && !s.tri_anti_only(t))
            .flat_map(|t| {
                let n = s.triangles[t].nodes;
                (0..3).map(move |k| (n[k].min(n[(k + 1) % 3]), n[k].max(n[(k + 1) % 3])))
            })
            .collect();
        e.sort();
        e.dedup();
        e
    };
    let mut prev_in: [Vec<bool>; 2] = [
        vec![false; w.vehicles[0].nodes.len()],
        vec![false; w.vehicles[1].nodes.len()],
    ];
    let mut prev_pos: [Vec<Vec3>; 2] = [
        w.vehicles[0]
            .nodes
            .iter()
            .map(|n| w.vehicles[0].origin + n.pos)
            .collect(),
        w.vehicles[1]
            .nodes
            .iter()
            .map(|n| w.vehicles[1].origin + n.pos)
            .collect(),
    ];
    let mut worst = [Dir::default(); 2];
    let (mut slab_max, mut slab_vol_max, mut slab_t) = (0.0f32, 0.0f32, 0.0f32);
    let mut entered: [HashMap<String, usize>; 2] = Default::default();
    let steps = (secs / 0.0005).round() as usize;
    let every = 8; // 4 ms
    println!("  t ms | A in B: nodes (deepest mm) face/gap, paint n p99/max mm, tyre n max, edges | B in A: same | slab max mm, volume L | dead tris A/B");
    for k in 0..=steps {
        if k > 0 {
            w.step();
        }
        if k % every != 0 {
            continue;
        }
        let full = k % (every * 5) == 0 || k == steps;
        let hulls = [hull(&w.vehicles[0]), hull(&w.vehicles[1])];
        let boxes = [aabb(&hulls[0]), aabb(&hulls[1])];
        let mut now = [Dir::default(); 2];
        let mut rend: [Vec<Vec3>; 2] = [Vec::new(), Vec::new()];
        if full {
            for c in 0..2 {
                let s = &w.vehicles[c];
                let n: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
                rend[c] = deform(&vis[c], &n)
                    .0
                    .into_iter()
                    .map(|q| q + s.origin)
                    .collect();
            }
        }
        for c in 0..2 {
            let o = 1 - c;
            let (x, y) = (&w.vehicles[c], &hulls[o]);
            let mut r = Dir::default();
            let mut ins = vec![false; x.nodes.len()];
            for &i in &x.contact_points {
                let p = x.origin + x.nodes[i as usize].pos;
                if !inside_box(p, boxes[o]) || winding(p, y) < 0.5 {
                    continue;
                }
                ins[i as usize] = true;
                r.nodes += 1;
                r.node_depth = r.node_depth.max(dist(p, y));
                if !prev_in[c][i as usize] {
                    let q = prev_pos[c][i as usize];
                    if y.iter().any(|t| seg_tri(q, p, t)) {
                        r.face += 1;
                    } else {
                        r.gap += 1;
                    }
                    *entered[c]
                        .entry(l.desc.nodes[i as usize].name.clone())
                        .or_default() += 1;
                }
            }
            prev_in[c] = ins;
            prev_pos[c] = x.nodes.iter().map(|n| x.origin + n.pos).collect();
            // Skin edges through the other's triangles.
            for (i, j) in edges_of(x) {
                let (p, q) = (
                    x.origin + x.nodes[i as usize].pos,
                    x.origin + x.nodes[j as usize].pos,
                );
                if !inside_box(p, boxes[o]) && !inside_box(q, boxes[o]) {
                    continue;
                }
                r.edges += y.iter().filter(|t| seg_tri(p, q, t)).count();
            }
            if full {
                let mut dp: Vec<f32> = Vec::new();
                for (q, &m) in rend[c].iter().zip(&vis[c].material) {
                    if (m != mat::PAINT && m != mat::TYRE) || !inside_box(*q, boxes[o]) {
                        continue;
                    }
                    if winding(*q, y) < 0.5 {
                        continue;
                    }
                    let dd = dist(*q, y);
                    if m == mat::TYRE {
                        r.tyre += 1;
                        r.tyre_depth = r.tyre_depth.max(dd);
                    } else {
                        r.paint += 1;
                        dp.push(dd);
                    }
                }
                r.paint_p99 = pct(&mut dp, 0.99);
                r.paint_depth = pct(&mut dp, 1.0);
            }
            now[c] = r;
            let wv = &mut worst[c];
            wv.nodes = wv.nodes.max(r.nodes);
            wv.node_depth = wv.node_depth.max(r.node_depth);
            wv.face += r.face;
            wv.gap += r.gap;
            wv.paint = wv.paint.max(r.paint);
            wv.paint_p99 = wv.paint_p99.max(r.paint_p99);
            wv.paint_depth = wv.paint_depth.max(r.paint_depth);
            wv.tyre = wv.tyre.max(r.tyre);
            wv.tyre_depth = wv.tyre_depth.max(r.tyre_depth);
            wv.edges = wv.edges.max(r.edges);
        }
        let (mut smax, mut svol) = (0.0f32, 0.0f32);
        if full {
            // Slab overlap: per 5 cm (x, y) bin, A's frontmost (max z) render
            // vertex against B's frontmost (min z).
            let bin = 0.05f32;
            let key = |q: Vec3| ((q.x / bin).floor() as i32, (q.y / bin).floor() as i32);
            let mut af: HashMap<(i32, i32), f32> = HashMap::new();
            let mut bf: HashMap<(i32, i32), f32> = HashMap::new();
            for (q, &m) in rend[0].iter().zip(&vis[0].material) {
                if m == mat::PAINT || m == mat::TYRE {
                    let e = af.entry(key(*q)).or_insert(f32::MIN);
                    *e = e.max(q.z);
                }
            }
            for (q, &m) in rend[1].iter().zip(&vis[1].material) {
                if m == mat::PAINT || m == mat::TYRE {
                    let e = bf.entry(key(*q)).or_insert(f32::MAX);
                    *e = e.min(q.z);
                }
            }
            for (kk, za) in &af {
                if let Some(zb) = bf.get(kk) {
                    let o = za - zb;
                    if o > 0.0 {
                        smax = smax.max(o);
                        svol += o * bin * bin;
                    }
                }
            }
            if smax > slab_max {
                slab_max = smax;
                slab_t = k as f32 * 0.0005;
            }
            slab_vol_max = slab_vol_max.max(svol);
        }
        let dead = |s: &ExplicitBeams| s.tri_dead.iter().filter(|&&x| x).count();
        if std::env::var("PITCH").is_ok() && full {
            // The front hubs' height (car A, B): a nose lifting or diving.
            let hy = |s: &ExplicitBeams| {
                ["hubFiL", "hubFiR"]
                    .iter()
                    .filter_map(|n| l.desc.nodes.iter().position(|x| x.name == *n))
                    .map(|i| s.origin.y + s.nodes[i].pos.y)
                    .sum::<f32>()
                    / 2.0
            };
            println!(
                "  {:4.0} ms front hubs y A {:.3} B {:.3} m",
                k as f32 * 0.5,
                hy(&w.vehicles[0]),
                hy(&w.vehicles[1])
            );
        }
        if full {
            let f = |r: &Dir| {
                format!(
                    "{:3} ({:4.0}) {:2}/{:2}, {:5} {:4.0}/{:4.0}, {:4} {:4.0}, {:3}",
                    r.nodes,
                    r.node_depth * 1e3,
                    r.face,
                    r.gap,
                    r.paint,
                    r.paint_p99 * 1e3,
                    r.paint_depth * 1e3,
                    r.tyre,
                    r.tyre_depth * 1e3,
                    r.edges
                )
            };
            println!(
                "  {:4.0} | {} | {} | {:4.0}, {:5.1} | {}/{}",
                k as f32 * 0.5,
                f(&now[0]),
                f(&now[1]),
                smax * 1e3,
                svol * 1e3,
                dead(&w.vehicles[0]),
                dead(&w.vehicles[1])
            );
        }
    }
    {
        let s = &w.vehicles[0];
        let dead: Vec<[String; 3]> = (0..s.triangles.len())
            .filter(|&t| s.tri_dead.get(t).copied().unwrap_or(false))
            .map(|t| {
                s.triangles[t]
                    .nodes
                    .map(|k| l.desc.nodes[k as usize].name.clone())
            })
            .collect();
        println!("  car A's dead triangles: {dead:?}");
    }
    for (c, name) in ["A in B", "B in A"].iter().enumerate() {
        let r = worst[c];
        println!(
            "  worst {name}: nodes {} (deepest {:.0} mm; entries through a face {}, through a gap {}), paint vertices {} (p99 {:.0}, deepest {:.0} mm), tyre vertices {} (deepest {:.0} mm), skin edges through triangles {}",
            r.nodes,
            r.node_depth * 1e3,
            r.face,
            r.gap,
            r.paint,
            r.paint_p99 * 1e3,
            r.paint_depth * 1e3,
            r.tyre,
            r.tyre_depth * 1e3,
            r.edges
        );
        let mut e: Vec<(&String, &usize)> = entered[c].iter().collect();
        e.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        println!(
            "    nodes that entered: {:?}",
            e.iter().take(40).collect::<Vec<_>>()
        );
    }
    println!(
        "  slab overlap: deepest {:.0} mm at {:.0} ms, largest volume {:.1} L",
        slab_max * 1e3,
        slab_t * 1e3,
        slab_vol_max * 1e3
    );
    Ok(())
}
