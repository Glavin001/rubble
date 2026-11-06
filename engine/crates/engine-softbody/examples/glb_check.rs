//! The vehicle art spec's automatic acceptance checks on a `.glb` (an
//! artist's delivery or gen2's `sedan.glb`): names, dimensions, wheels,
//! manifoldness, panel edge length and triangle shape, shut-line gaps,
//! interpenetration, triangle budget, materials and UV overlap.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example glb_check -- [path.glb]
//! ```
//! The path defaults to `engine/crates/engine-webgpu/__mods/gen2/sedan.glb`.
//! Exits non-zero when a check fails. (The Khronos glTF Validator and the
//! rest-fidelity check after binding are separate.)
use engine_softbody::glb::{self, Glb};
use glam::Vec3;
use std::collections::HashMap;

const MATERIALS: [&str; 10] = [
    "paint",
    "chrome",
    "trim_black",
    "rubber",
    "glass",
    "lamp_lens",
    "lamp_emissive",
    "interior_fabric",
    "interior_plastic",
    "metal_bare",
];

fn required_parts() -> Vec<String> {
    let mut v: Vec<String> = [
        "body_shell",
        "hood",
        "trunk_lid",
        "fender_FL",
        "fender_FR",
        "bumper_front",
        "bumper_rear",
        "grille",
        "glass_windshield",
        "glass_rear",
        "glass_quarter_L",
        "glass_quarter_R",
        "mirror_L",
        "mirror_R",
        "wiper_L",
        "wiper_R",
        "dash",
        "steering_wheel",
        "seat_FL",
        "seat_FR",
        "seat_rear",
        "console",
        "headliner",
        "engine_block",
        "radiator",
        "firewall",
        "underbody",
        "exhaust",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for w in ["FL", "FR", "RL", "RR"] {
        for p in [
            "door",
            "glass_door",
            "handle",
            "rim",
            "tyre",
            "brake_disc",
            "door_trim",
            "wheel_well",
        ] {
            v.push(format!("{p}_{w}"));
        }
    }
    for s in ["L", "R"] {
        for l in ["headlight", "taillight"] {
            v.push(format!("{l}_{s}_lens"));
            v.push(format!("{l}_{s}_housing"));
        }
    }
    v
}

fn required_markers() -> Vec<String> {
    let mut v = Vec::new();
    for w in ["FL", "FR", "RL", "RR"] {
        v.push(format!("wheel_center_{w}"));
        v.push(format!("hinge_door_{w}_upper"));
        v.push(format!("hinge_door_{w}_lower"));
        v.push(format!("latch_door_{w}"));
    }
    for s in ["L", "R"] {
        v.push(format!("hinge_hood_{s}"));
        v.push(format!("hinge_trunk_{s}"));
        v.push(format!("mount_mirror_{s}"));
    }
    v.extend(
        [
            "latch_hood",
            "latch_trunk",
            "seat_H_point_FL",
            "seat_H_point_FR",
        ]
        .map(String::from),
    );
    v
}

fn is_panel(name: &str) -> bool {
    [
        "body_shell",
        "hood",
        "trunk_lid",
        "door_F",
        "door_R",
        "fender_",
        "bumper_",
    ]
    .iter()
    .any(|p| name.starts_with(p))
}

/// (positions, triangles, triangle material names) of a part, all primitives.
fn soup(
    g: &Glb,
    name: &str,
) -> (
    Vec<Vec3>,
    Vec<Vec3>,
    Vec<[usize; 3]>,
    Vec<String>,
    Vec<[f32; 2]>,
) {
    let mut p = Vec::new();
    let mut n = Vec::new();
    let mut uv = Vec::new();
    let mut t = Vec::new();
    let mut tm = Vec::new();
    for m in g.meshes.iter().filter(|m| m.name == name) {
        for pr in &m.prims {
            let base = p.len();
            p.extend(pr.pos.iter().map(|q| Vec3::from_array(*q)));
            n.extend(pr.nrm.iter().map(|q| Vec3::from_array(*q)));
            uv.extend(pr.uv.iter().copied());
            let mn = g
                .materials
                .get(pr.material)
                .map_or("?".to_string(), |x| x.name.clone());
            for c in pr.idx.chunks(3) {
                t.push([
                    base + c[0] as usize,
                    base + c[1] as usize,
                    base + c[2] as usize,
                ]);
                tm.push(mn.clone());
            }
        }
    }
    (p, n, t, tm, uv)
}

fn pct(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * q).round() as usize]
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../engine-webgpu/__mods/gen2/sedan.glb"
        )
        .to_string()
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let g = glb::read(&bytes).expect("glb");
    let mut fails = 0usize;
    let mut row = |name: &str, ok: bool, detail: String| {
        if !ok {
            fails += 1;
        }
        println!(
            "| {name} | {} | {detail} |",
            if ok { "pass" } else { "FAIL" }
        );
    };
    println!("{path}\n\n| check | result | detail |\n|---|---|---|");
    // Hierarchy and names.
    row(
        "root named `vehicle`",
        g.root == "vehicle",
        format!("root `{}`", g.root),
    );
    let names: Vec<&str> = g.meshes.iter().map(|m| m.name.as_str()).collect();
    let missing: Vec<String> = required_parts()
        .into_iter()
        .filter(|p| !names.contains(&p.as_str()))
        .collect();
    row(
        "every part present",
        missing.is_empty(),
        format!("{} parts; missing {:?}", names.len(), missing),
    );
    let mk: HashMap<&str, Vec3> = g
        .markers
        .iter()
        .map(|k| (k.name.as_str(), Vec3::from_array(k.pos)))
        .collect();
    let missing: Vec<String> = required_markers()
        .into_iter()
        .filter(|p| !mk.contains_key(p.as_str()))
        .collect();
    let bolts_f = g
        .markers
        .iter()
        .filter(|k| k.name.starts_with("bolt_fender_FL_"))
        .count();
    let bolts_b = g
        .markers
        .iter()
        .filter(|k| k.name.starts_with("bolt_bumper_front_"))
        .count();
    row(
        "every marker present",
        missing.is_empty() && (3..=5).contains(&bolts_f) && (4..=6).contains(&bolts_b),
        format!(
            "{} markers; missing {:?}; fender bolts {bolts_f}, bumper bolts {bolts_b}",
            g.markers.len(),
            missing
        ),
    );
    // Materials.
    let bad: Vec<&str> = g
        .materials
        .iter()
        .map(|m| m.name.as_str())
        .filter(|n| !MATERIALS.contains(n))
        .collect();
    row(
        "materials use only the spec's names",
        bad.is_empty(),
        format!("{} materials; others {:?}", g.materials.len(), bad),
    );
    let glass_ok = g
        .materials
        .iter()
        .filter(|m| m.name == "glass")
        .all(|m| m.roughness <= 0.05 && (m.blend || m.base[3] < 1.0));
    row(
        "glass transparent at roughness ≤ 0.05",
        glass_ok,
        String::new(),
    );
    // Dimensions.
    let bbox = |f: &dyn Fn(&str) -> bool| {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for m in g.meshes.iter().filter(|m| f(&m.name)) {
            for p in m.prims.iter().flat_map(|p| p.pos.iter()) {
                lo = lo.min(Vec3::from_array(*p));
                hi = hi.max(Vec3::from_array(*p));
            }
        }
        (lo, hi)
    };
    let (lo, hi) = bbox(&|n| !n.starts_with("mirror") && !n.starts_with("exhaust"));
    let wc = |w: &str| {
        mk.get(format!("wheel_center_{w}").as_str())
            .copied()
            .unwrap_or(Vec3::NAN)
    };
    let wb = (wc("FL").z + wc("FR").z) / 2.0 - (wc("RL").z + wc("RR").z) / 2.0;
    let track_f = wc("FL").x - wc("FR").x;
    let track_r = wc("RL").x - wc("RR").x;
    let (blo, _) = bbox(&|n| {
        !["tyre", "rim", "brake", "wheel_well", "exhaust"]
            .iter()
            .any(|p| n.starts_with(p))
    });
    let tyre = |w: &str| bbox(&|n| n == format!("tyre_{w}"));
    let mut dims = Vec::new();
    let mut dim = |label: &str, v: f32, target: f32, tol: f32| {
        let ok = (v - target).abs() <= tol;
        dims.push(format!(
            "{label} {:.0} mm ({:.0} ± {:.0}){}",
            v * 1e3,
            target * 1e3,
            tol * 1e3,
            if ok { "" } else { " ✗" }
        ));
        ok
    };
    let mut all = true;
    all &= dim("length", hi.z - lo.z, 4.880, 0.050);
    all &= dim("width", hi.x - lo.x, 1.860, 0.030);
    all &= dim("height", hi.y - lo.y.min(0.0), 1.450, 0.030);
    all &= dim("wheelbase", wb, 2.830, 0.020);
    all &= dim("track F", track_f, 1.600, 0.020);
    all &= dim("track R", track_r, 1.600, 0.020);
    all &= dim(
        "front overhang",
        hi.z - (wc("FL").z + wc("FR").z) / 2.0,
        0.970,
        0.050,
    );
    all &= dim("ground clearance", blo.y, 0.140, 0.015);
    let (tl, th) = tyre("FL");
    all &= dim("tyre diameter", th.y - tl.y, 0.683, 0.005);
    row("dimensions within tolerance", all, dims.join("; "));
    // Origin and stance.
    let mid = (wc("FL") + wc("FR") + wc("RL") + wc("RR")) / 4.0;
    let mut stance = Vec::new();
    let mut st_ok = mid.z.abs() < 0.005 && mid.x.abs() < 0.005;
    for w in ["FL", "FR", "RL", "RR"] {
        let (a, b) = tyre(w);
        let c = (a + b) / 2.0;
        let off = (c - wc(w)).length();
        st_ok &= a.y.abs() < 0.002 && off < 0.005;
        stance.push(format!(
            "{w}: bottom {:.1} mm, centre {:.1} mm off",
            a.y * 1e3,
            off * 1e3
        ));
    }
    row(
        "origin between the axles; tyres on y = 0, on their markers",
        st_ok,
        format!(
            "axle midpoint ({:.1}, {:.1}) mm; {}",
            mid.x * 1e3,
            mid.z * 1e3,
            stance.join("; ")
        ),
    );
    // Per part: manifold, zero-area, inverted normals; panels: edges, aspect.
    let mut nonmanifold = 0;
    let mut zero = 0;
    let mut inverted = 0;
    let mut edges_panel: Vec<f32> = Vec::new();
    let mut aspect_panel: Vec<f32> = Vec::new();
    let mut worst_aspect = (0.0f32, String::new());
    let mut over6: HashMap<String, usize> = HashMap::new();
    let mut groups: HashMap<&str, usize> = HashMap::new();
    let mut tris_total = 0;
    for m in &g.meshes {
        let (p, n, t, _, _) = soup(&g, &m.name);
        tris_total += t.len();
        let grp = if m.name.starts_with("rim")
            || m.name.starts_with("tyre")
            || m.name.starts_with("brake")
        {
            "wheel"
        } else if is_panel(&m.name) {
            "body"
        } else if [
            "dash",
            "steering",
            "seat",
            "console",
            "door_trim",
            "headliner",
        ]
        .iter()
        .any(|k| m.name.starts_with(k))
        {
            "interior"
        } else {
            "other"
        };
        *groups.entry(grp).or_insert(0) += t.len();
        // Weld by position.
        let key = |q: Vec3| {
            (
                (q.x * 1e5).round() as i64,
                (q.y * 1e5).round() as i64,
                (q.z * 1e5).round() as i64,
            )
        };
        let mut weld: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let id: Vec<usize> = p
            .iter()
            .map(|&q| {
                let l = weld.len();
                *weld.entry(key(q)).or_insert(l)
            })
            .collect();
        let mut ec: HashMap<(usize, usize), usize> = HashMap::new();
        for tr in &t {
            let w = tr.map(|k| id[k]);
            for e in 0..3 {
                let (a, b) = (w[e], w[(e + 1) % 3]);
                if a != b {
                    *ec.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
            let (a, b, c) = (p[tr[0]], p[tr[1]], p[tr[2]]);
            let cr = (b - a).cross(c - a);
            let area = 0.5 * cr.length();
            if area < 1e-9 {
                zero += 1;
                continue;
            }
            if cr.dot(n[tr[0]] + n[tr[1]] + n[tr[2]]) < 0.0 {
                inverted += 1;
                if std::env::var("GLB_DEBUG").is_ok() {
                    println!("  inverted {} ({:.3}, {:.3}, {:.3})", m.name, a.x, a.y, a.z);
                }
            }
            if is_panel(&m.name) {
                let l = [(b - a).length(), (c - b).length(), (a - c).length()];
                edges_panel.extend(l);
                let lmax = l.iter().copied().fold(0.0, f32::max);
                let asp = lmax * lmax / (2.0 * area);
                aspect_panel.push(asp);
                if asp > 6.0 {
                    if std::env::var("GLB_DEBUG").is_ok()
                        && asp
                            > std::env::var("GLB_DEBUG")
                                .unwrap()
                                .parse::<f32>()
                                .unwrap_or(30.0)
                    {
                        println!("  aspect {asp:.0} {} ({:.3}, {:.3}, {:.3}) ({:.3}, {:.3}, {:.3}) ({:.3}, {:.3}, {:.3})", m.name, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
                    }
                    *over6.entry(m.name.clone()).or_insert(0) += 1;
                }
                if asp > worst_aspect.0 {
                    worst_aspect = (
                        asp,
                        format!("{} at ({:.3}, {:.3}, {:.3})", m.name, a.x, a.y, a.z),
                    );
                }
            }
        }
        // Parts may hold several pieces that share positions (an inner panel
        // and the skin's hem flange do not): count edges on > 2 triangles.
        nonmanifold += ec.values().filter(|&&c| c > 2).count();
    }
    row(
        "no non-manifold edges",
        nonmanifold == 0,
        format!("{nonmanifold} edges on more than two triangles"),
    );
    row("no zero-area triangles", zero == 0, format!("{zero}"));
    row(
        "no inverted normals",
        inverted == 0,
        format!("{inverted} triangles facing against their vertex normals"),
    );
    let e95 = pct(&mut edges_panel.clone(), 0.95);
    let e50 = pct(&mut edges_panel.clone(), 0.5);
    row(
        "panel edge p95 ≤ 30 mm",
        e95 <= 0.030,
        format!("p50 {:.1} mm, p95 {:.1} mm", e50 * 1e3, e95 * 1e3),
    );
    let a99 = pct(&mut aspect_panel.clone(), 0.99);
    let amax = aspect_panel.iter().copied().fold(0.0, f32::max);
    let mut o6: Vec<(String, usize)> = over6.into_iter().collect();
    o6.sort_by(|a, b| b.1.cmp(&a.1));
    row(
        "panel triangle aspect ≤ 6:1",
        amax <= 6.0,
        format!(
            "p99 {a99:.2}, max {amax:.2} ({}); over 6: {:?}",
            worst_aspect.1, o6
        ),
    );
    // Budget.
    let gb = |k: &str| groups.get(k).copied().unwrap_or(0);
    let wheel_max = ["FL", "FR", "RL", "RR"]
        .iter()
        .map(|w| {
            ["rim", "tyre", "brake_disc"]
                .iter()
                .map(|p| soup(&g, &format!("{p}_{w}")).2.len())
                .sum::<usize>()
        })
        .max()
        .unwrap_or(0);
    let ok = (150_000..=300_000).contains(&gb("body"))
        && wheel_max <= 15_000
        && gb("interior") <= 80_000
        && gb("other") <= 60_000
        && tris_total <= 500_000;
    row(
        "triangle budget",
        ok,
        format!("body {} (150k–300k), wheel max {wheel_max} (≤ 15k), interior {} (≤ 80k), lamps/glass/trim/bay {} (≤ 60k), total {tris_total} (≤ 500k)", gb("body"), gb("interior"), gb("other")),
    );
    // Shut-line gaps: each panel's border vertices to the nearest vertex of
    // another panel part within 15 mm.
    let panels: Vec<&str> = names.iter().copied().filter(|n| is_panel(n)).collect();
    // Every panel triangle in a 2 cm grid (by its bounding box).
    let soups: Vec<_> = panels.iter().map(|n| soup(&g, n)).collect();
    // GLB_DEBUG_AT=x,y,z: list the panel vertices within 15 mm of a point.
    if let Ok(at) = std::env::var("GLB_DEBUG_AT") {
        let v: Vec<f32> = at.split(',').filter_map(|s| s.parse().ok()).collect();
        let c = Vec3::new(v[0], v[1], v[2]);
        for (pi, s) in soups.iter().enumerate() {
            for (k, q) in s.0.iter().enumerate() {
                if q.distance(c) < 0.015 {
                    println!(
                        "  {} v{k} ({:.4}, {:.4}, {:.4}) n ({:.2}, {:.2}, {:.2})",
                        panels[pi], q.x, q.y, q.z, s.1[k].x, s.1[k].y, s.1[k].z
                    );
                }
            }
        }
    }
    let cell = 0.02f32;
    let ck = |q: Vec3| {
        (
            (q.x / cell).floor() as i32,
            (q.y / cell).floor() as i32,
            (q.z / cell).floor() as i32,
        )
    };
    let mut tgrid: HashMap<(i32, i32, i32), Vec<(usize, usize)>> = HashMap::new();
    for (pi, s) in soups.iter().enumerate() {
        for (ti, tr) in s.2.iter().enumerate() {
            let (a, b, c) = (s.0[tr[0]], s.0[tr[1]], s.0[tr[2]]);
            let (lo, hi) = (ck(a.min(b).min(c)), ck(a.max(b).max(c)));
            for x in lo.0..=hi.0 {
                for y in lo.1..=hi.1 {
                    for z in lo.2..=hi.2 {
                        tgrid.entry((x, y, z)).or_default().push((pi, ti));
                    }
                }
            }
        }
    }
    let closest = |p: Vec3, a: Vec3, b: Vec3, c: Vec3| -> Vec3 {
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
    };
    // Each panel vertex's distance to the nearest other panel; the gap of a
    // shut line is the least of these round each 2 cm of it. A vertex behind
    // another panel's outer face (its triangle's normal) by over 1 mm, within
    // that triangle, interpenetrates.
    let mut cell_min: HashMap<((i32, i32, i32), usize, usize), f32> = HashMap::new();
    let mut pen_max = (0.0f32, String::new());
    for (pi, s) in soups.iter().enumerate() {
        for (vi, &q) in s.0.iter().enumerate() {
            let c = ck(q);
            let mut best = (f32::MAX, usize::MAX);
            for &(pj, tj) in tgrid.get(&c).map(|v| v.as_slice()).unwrap_or(&[]) {
                if pj == pi {
                    continue;
                }
                let o = &soups[pj];
                let tr = o.2[tj];
                let (a, b, cc) = (o.0[tr[0]], o.0[tr[1]], o.0[tr[2]]);
                let x = closest(q, a, b, cc);
                let d = q.distance(x);
                if d < best.0 {
                    best = (d, pj);
                }
                // Penetration: inside the triangle (not at its border), behind it.
                let nrm = (b - a).cross(cc - a).normalize_or_zero();
                let into = -(q - x).dot(nrm);
                let on_face = (q - nrm * (q - a).dot(nrm)).distance(x) < 1e-5;
                if on_face
                    && into > 0.001
                    && into < 0.01
                    && s.1[vi].dot(nrm) > 0.5
                    && into > pen_max.0
                {
                    pen_max = (
                        into,
                        format!(
                            "{} into {} at ({:.3}, {:.3}, {:.3})",
                            panels[pi], panels[pj], q.x, q.y, q.z
                        ),
                    );
                }
            }
            if best.0 < 0.015 {
                let key = (ck(q), pi.min(best.1), pi.max(best.1));
                let e = cell_min.entry(key).or_insert(f32::MAX);
                *e = e.min(best.0);
            }
        }
    }
    let mut gaps: Vec<f32> = cell_min.values().copied().collect();
    let (g05, g50, g95) = (
        pct(&mut gaps.clone(), 0.05),
        pct(&mut gaps.clone(), 0.5),
        pct(&mut gaps.clone(), 0.95),
    );
    let gmin = gaps.iter().copied().fold(f32::MAX, f32::min);
    let mut tight: HashMap<String, usize> = HashMap::new();
    for (&(c, a, b), &d) in &cell_min {
        if std::env::var("GLB_DEBUG").is_ok() && d > 0.006 {
            println!(
                "  wide {:.1} mm {}/{} ({:.2}, {:.2}, {:.2})",
                d * 1e3,
                panels[a],
                panels[b],
                c.0 as f32 * cell,
                c.1 as f32 * cell,
                c.2 as f32 * cell
            );
        }
        if d < 0.003 {
            let at = format!(
                "{}/{} e.g. ({:.2}, {:.2}, {:.2})",
                panels[a],
                panels[b],
                c.0 as f32 * cell,
                c.1 as f32 * cell,
                c.2 as f32 * cell
            );
            let key = tight
                .keys()
                .find(|k| k.starts_with(&format!("{}/{} ", panels[a], panels[b])))
                .cloned()
                .unwrap_or(at);
            *tight.entry(key).or_insert(0) += 1;
        }
    }
    let mut tight: Vec<(String, usize)> = tight.into_iter().collect();
    tight.sort_by(|a, b| b.1.cmp(&a.1));
    tight.truncate(8);
    row(
        "shut-line gaps 3–6 mm",
        g05 >= 0.003 && g95 <= 0.006,
        format!(
            "{} shut-line cells (2 cm): min {:.1}, p5 {:.1}, p50 {:.1}, p95 {:.1} mm; under 3 mm: {:?}",
            gaps.len(),
            gmin * 1e3,
            g05 * 1e3,
            g50 * 1e3,
            g95 * 1e3,
            tight
        ),
    );
    row(
        "no interpenetration over 1 mm (panels)",
        pen_max.0 <= 0.001,
        format!("deepest {:.1} mm {}", pen_max.0 * 1e3, pen_max.1),
    );
    // UV0 overlap on painted panels: rasterise at 2048².
    let res = 2048usize;
    let mut cover = vec![0u16; res * res];
    let mut opairs: HashMap<(usize, usize), usize> = HashMap::new();
    let mut overl = 0usize;
    let mut covered = 0usize;
    for (pi, s) in soups.iter().enumerate() {
        for (tr, mn) in s.2.iter().zip(&s.3) {
            if mn != "paint" {
                continue;
            }
            let uv = tr.map(|k| s.4[k]);
            let to = |u: [f32; 2]| (u[0] * res as f32, u[1] * res as f32);
            let (a, b, c) = (to(uv[0]), to(uv[1]), to(uv[2]));
            let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
            if area.abs() < 1e-6 {
                continue;
            }
            let (x0, x1) = (
                a.0.min(b.0).min(c.0).floor().max(0.0) as usize,
                a.0.max(b.0).max(c.0).ceil().min(res as f32 - 1.0) as usize,
            );
            let (y0, y1) = (
                a.1.min(b.1).min(c.1).floor().max(0.0) as usize,
                a.1.max(b.1).max(c.1).ceil().min(res as f32 - 1.0) as usize,
            );
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let e = |p: (f32, f32), q: (f32, f32)| {
                        (q.0 - p.0) * (py - p.1) - (q.1 - p.1) * (px - p.0)
                    };
                    let (w0, w1, w2) = (e(b, c), e(c, a), e(a, b));
                    let inside = if area > 0.0 {
                        w0 > 0.0 && w1 > 0.0 && w2 > 0.0
                    } else {
                        w0 < 0.0 && w1 < 0.0 && w2 < 0.0
                    };
                    if inside {
                        let k = y * res + x;
                        if cover[k] > 0 {
                            overl += 1;
                            *opairs.entry((cover[k] as usize - 1, pi)).or_insert(0) += 1;
                        } else {
                            covered += 1;
                        }
                        cover[k] = pi as u16 + 1;
                    }
                }
            }
        }
    }
    let inside_unit = soups.iter().all(|s| {
        s.4.iter()
            .all(|u| (0.0..=1.0).contains(&u[0]) && (0.0..=1.0).contains(&u[1]))
    });
    row(
        "UV0 non-overlapping on painted panels",
        overl == 0 && inside_unit,
        format!("{covered} texels covered at 2048², {overl} covered twice ({:?}); all in [0, 1]: {inside_unit}", { let mut v: Vec<_> = opairs.iter().map(|(&(a, b), &c)| (panels[a], panels[b], c)).collect(); v.sort_by(|a, b| b.2.cmp(&a.2)); v.truncate(6); v }),
    );
    println!("\n{} check(s) failed", fails);
    std::process::exit(if fails > 0 { 1 } else { 0 });
}
