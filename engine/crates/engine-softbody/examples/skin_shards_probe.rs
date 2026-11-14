//! Diagnostic: the showcase sedan's visual skin after crashes, as the page
//! `softbody.html?solver=explicit&car=showcase` (drive yard, raycast wheels)
//! draws it. CPU only. Per scene it reports, per frame and at the end:
//! flung nodes (far past the car's rest radius from its centroid) with their
//! beams and speed history; the skin's visible edge stretch (the renderer's
//! tear field emulated: a triangle's part where the interpolated tear is
//! under ½ is drawn); flipped triangles; and attributes every shard.
//!
//! `cargo run --release -p engine-softbody --example skin_shards_probe -- <scene> [km/h]`
//! scenes: wall, wall2 (two hits), side (a car into the rear door), all.
//! `OBJ=dir`: writes the deformed skin (drawn triangles) as `.obj` files.
use engine_softbody::authoring::skin::{
    deform_beams, raycast_visual, showcase_visual_lod, showcase_visual_triangles_lod, unit_states,
    vertex_tears, BeamState, ShowcaseVisual,
};
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{Quat, Vec3};

fn settings() -> ExplicitBeamsSettings {
    // The page's: drag and turbulence on, fast inverse square root.
    ExplicitBeamsSettings {
        drag: std::env::var("DRAG").map_or(true, |v| v != "0"),
        turbulence: std::env::var("DRAG").map_or(true, |v| v != "0"),
        ..Default::default()
    }
}

/// A raycast spawn at `at` (x, z: its centre), yaw degrees, moving at `vel`:
/// its lowest wheel's bottom 2 cm up, as `ExplicitSim::spawn`.
fn spawn(b: &Built, at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let (p, _) =
        ExplicitBeams::from_truck_raycast(&b.truck, None, Vec3::ZERO, yaw, Vec3::ZERO, settings())
            .unwrap();
    let d = p.drive.as_ref().unwrap();
    let low = d
        .config
        .wheels
        .iter()
        .map(|w| 0.5 * (p.nodes[w.node1 as usize].pos.y + p.nodes[w.node2 as usize].pos.y) - w.radius)
        .fold(f32::MAX, f32::min);
    let (lo, hi) = p
        .nodes
        .iter()
        .fold((Vec3::MAX, Vec3::MIN), |(a, c), n| (a.min(n.pos), c.max(n.pos)));
    let mid = 0.5 * (lo + hi);
    let pos = Vec3::new(at.x - mid.x, 0.02 - low, at.z - mid.z);
    let (mut s, _) =
        ExplicitBeams::from_truck_raycast(&b.truck, None, pos, yaw, vel, settings()).unwrap();
    // Wheels spinning at the ground speed.
    let sp = vel.length();
    if let Some(d) = s.drive.as_mut() {
        let r: Vec<f32> = d.config.wheels.iter().map(|w| w.radius).collect();
        for (w, r) in d.wheels.iter_mut().zip(r) {
            w.omega = sp / r;
        }
    }
    s
}

fn states(s: &ExplicitBeams) -> Vec<BeamState> {
    s.beams
        .iter()
        .map(|b| BeamState {
            broken: b.broken,
            rest: b.rest,
        })
        .collect()
}

/// Union-find components of the nodes over live beams.
fn components(s: &ExplicitBeams) -> Vec<u32> {
    let n = s.nodes.len();
    let mut p: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for b in s.beams.iter().filter(|b| !b.broken) {
        let (x, y) = (find(&mut p, b.a), find(&mut p, b.b));
        if x != y {
            p[x as usize] = y;
        }
    }
    (0..n as u32).map(|i| find(&mut p, i)).collect()
}

/// The drawn length of an edge (its part where the linear tear is under ½).
fn drawn(len: f32, ta: f32, tb: f32) -> f32 {
    match (ta < 0.5, tb < 0.5) {
        (true, true) => len,
        (false, false) => 0.0,
        _ => {
            let (lo, hi) = (ta.min(tb), ta.max(tb));
            len * (0.5 - lo) / (hi - lo).max(1e-9)
        }
    }
}

#[derive(Default, Debug)]
struct SkinStats {
    tris: usize,
    drawn: usize,
    /// Drawn triangles with a visible edge past 3× its rest length and 5 cm.
    shards: usize,
    long_m: f32,
    max_stretch: f32,
    p99_stretch: f32,
    flipped: usize,
    /// Shard causes: a vertex bound to a flung node; vertices bound to nodes in
    /// different live-beam components (a tear spanned); the rest (lattice).
    by_flung: usize,
    by_split: usize,
    by_lattice: usize,
    /// Of the lattice shards: vertices on different parts (a seam).
    lattice_seam: usize,
    /// Vertices whose bound nodes span components.
    split_vertices: usize,
    /// Vertices whose displacement departs from its mesh neighbours' mean by
    /// more than 3× its mean rest edge (saw-tooth, spikes).
    jumpy: usize,
    samples: Vec<([usize; 3], f32, f32)>,
}

fn skin_stats(
    v: &ShowcaseVisual,
    s: &ExplicitBeams,
    flung: &[bool],
    obj: Option<&str>,
) -> SkinStats {
    let nodes = s.positions();
    let bs = states(s);
    let us = unit_states(v, &bs);
    let (pos, _nrm) = deform_beams(v, &nodes, Some(&us), Some(&bs));
    let vt = vertex_tears(v, &nodes, Some(&bs), &pos);
    let comp = components(s);
    let rp = |i: u32| Vec3::from_array(v.positions[i as usize]);
    let mut st = SkinStats::default();
    let mut stretches = Vec::new();
    let n = v.positions.len();
    let mut nb_sum = vec![Vec3::ZERO; n];
    let mut nb_cnt = vec![0u32; n];
    let mut edge_sum = vec![0f32; n];
    let mut out = String::new();
    let bound_flung = |i: usize| v.rigid[i] < 2 && v.bind_nodes[i].iter().any(|&k| flung[k as usize]);
    let split = |i: usize| {
        if v.rigid[i] >= 2 {
            return false;
        }
        let c = v.bind_nodes[i].map(|k| comp[k as usize]);
        c[0] != c[1] || c[1] != c[2]
    };
    st.split_vertices = (0..n).filter(|&i| split(i)).count();
    for t in v.indices.chunks_exact(3) {
        st.tris += 1;
        let ids = [t[0] as usize, t[1] as usize, t[2] as usize];
        let p = ids.map(|i| pos[i]);
        if p.iter().any(|q| !q.x.is_finite()) {
            continue;
        }
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let d = (p[b] - p[a]) - (rp(t[b]) - rp(t[a]));
            let _ = d;
            nb_sum[ids[a]] += p[b] - rp(t[b]);
            nb_cnt[ids[a]] += 1;
            nb_sum[ids[b]] += p[a] - rp(t[a]);
            nb_cnt[ids[b]] += 1;
            let l0 = rp(t[a]).distance(rp(t[b]));
            edge_sum[ids[a]] += l0;
            edge_sum[ids[b]] += l0;
        }
        let tv = ids.map(|i| vt[i]);
        if tv.iter().all(|&x| x >= 0.5) {
            continue;
        }
        st.drawn += 1;
        let mut worst = 0.0f32;
        let mut wl = 0.0f32;
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let l0 = rp(t[a]).distance(rp(t[b])).max(1e-4);
            let l = drawn(p[a].distance(p[b]), tv[a], tv[b]);
            worst = worst.max(l / l0);
            wl = wl.max(l);
        }
        stretches.push(worst);
        // Flipped: the face normal against its rest orientation carried by
        // its vertices' skinned normals.
        let fn_now = (p[1] - p[0]).cross(p[2] - p[0]);
        let r = ids.map(|i| Vec3::from_array(v.positions[i]));
        let fn_rest = (r[1] - r[0]).cross(r[2] - r[0]);
        let vn: Vec3 = ids.iter().map(|&i| Vec3::from_array(v.normals[i])).sum();
        if fn_rest.dot(vn) > 0.0 && fn_now.length() > 1e-9 && {
            // The vertices' current normals.
            let now: Vec3 = ids.iter().map(|&i| _nrm[i]).sum();
            fn_now.dot(now) < 0.0
        } {
            st.flipped += 1;
        }
        if worst > 3.0 && wl > 0.05 {
            st.shards += 1;
            st.samples.push((ids, worst, wl));
            st.long_m = st.long_m.max(wl);
            if ids.iter().any(|&i| bound_flung(i)) {
                st.by_flung += 1;
            } else if ids.iter().any(|&i| split(i))
                || {
                    let c: Vec<u32> = ids
                        .iter()
                        .filter(|&&i| v.rigid[i] < 2)
                        .map(|&i| comp[v.bind_nodes[i][0] as usize])
                        .collect();
                    c.windows(2).any(|w| w[0] != w[1])
                }
            {
                st.by_split += 1;
            } else {
                st.by_lattice += 1;
                if v.part[ids[0]] != v.part[ids[1]] || v.part[ids[1]] != v.part[ids[2]] {
                    st.lattice_seam += 1;
                }
            }
        }
        if obj.is_some() {
            out.push_str(&format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1));
        }
    }
    for i in 0..n {
        if nb_cnt[i] == 0 || !pos[i].x.is_finite() {
            continue;
        }
        let me = pos[i] - rp(i as u32);
        let mean = nb_sum[i] / nb_cnt[i] as f32;
        let e = edge_sum[i] / nb_cnt[i] as f32;
        if mean.is_finite() && (me - mean).length() > 3.0 * e.max(0.005) {
            st.jumpy += 1;
        }
    }
    stretches.sort_by(f32::total_cmp);
    if !stretches.is_empty() {
        st.max_stretch = *stretches.last().unwrap();
        st.p99_stretch = stretches[(stretches.len() as f32 * 0.99) as usize];
    }
    if std::env::var("DUMP").is_ok() && !st.samples.is_empty() {
        let piece = |i: usize| {
            v.pieces
                .iter()
                .find(|p| (p.1 as usize..(p.1 + p.2) as usize).contains(&i))
                .map(|p| p.0.clone())
                .unwrap_or_default()
        };
        let mut h: std::collections::BTreeMap<String, usize> = Default::default();
        for (ids, _, _) in &st.samples {
            *h.entry(piece(ids[0])).or_default() += 1;
        }
        let mut hv: Vec<_> = h.into_iter().collect();
        hv.sort_by_key(|x| std::cmp::Reverse(x.1));
        eprintln!("  shards by piece: {:?}", &hv[..hv.len().min(20)]);
        // Binding census: inside the simplex, on a skin triangle.
        let skin: std::collections::HashSet<[u32; 3]> = v.normal_tris.iter().map(|t| { let mut u = *t; u.sort(); u }).collect();
        let census = |ids: &mut dyn Iterator<Item = usize>| {
            let (mut n, mut out, mut on_skin, mut far) = (0, 0, 0, 0);
            for i in ids {
                if v.rigid[i] != 0 { continue; }
                n += 1;
                let [a, b, h] = v.bind_coords[i];
                if a < -0.02 || b < -0.02 || a + b > 1.02 { out += 1; }
                let mut u = v.bind_nodes[i]; u.sort();
                if skin.contains(&u) { on_skin += 1; }
                if h.abs() > 0.03 { far += 1; }
            }
            format!("{n} affine vertices: {:.1} % outside their triangle, {:.1} % on a skin triangle, {:.1} % |h| > 3 cm", 100.0 * out as f32 / n as f32, 100.0 * on_skin as f32 / n as f32, 100.0 * far as f32 / n as f32)
        };
        eprintln!("  all: {}", census(&mut (0..n)));
        let mut sv: Vec<usize> = st.samples.iter().flat_map(|s| s.0).collect();
        sv.sort(); sv.dedup();
        eprintln!("  shard vertices: {}", census(&mut sv.iter().copied()));
        // Shard triangles whose three vertices share one binding triangle.
        let same = st.samples.iter().filter(|s| { let t = s.0.map(|i| { let mut u = v.bind_nodes[i]; u.sort(); u }); t[0] == t[1] && t[1] == t[2] }).count();
        eprintln!("  shard triangles on one binding triangle: {same} of {}", st.samples.len());
        // Bind offsets of shard vertices against all vertices.
        let mut hs: Vec<f32> = st.samples.iter().flat_map(|s| s.0).map(|i| v.bind_coords[i][2].abs()).collect();
        hs.sort_by(f32::total_cmp);
        let mut ha: Vec<f32> = (0..n).filter(|&i| v.rigid[i] == 0).map(|i| v.bind_coords[i][2].abs()).collect();
        ha.sort_by(f32::total_cmp);
        let q = |v: &Vec<f32>, f: f32| v[((v.len() - 1) as f32 * f) as usize];
        eprintln!(
            "  |h| shard vertices median {:.3} p90 {:.3}; all vertices median {:.3} p90 {:.3} max {:.3}",
            q(&hs, 0.5), q(&hs, 0.9), q(&ha, 0.5), q(&ha, 0.9), q(&ha, 1.0)
        );
        let mut sk_st = Vec::new();
        for (ids, worst, wl) in st.samples.iter().step_by((st.samples.len() / 6).max(1)).take(6) {
            // The skeleton's own stretch over the bound nodes.
            let mut nn: Vec<u32> = ids.iter().flat_map(|&i| v.bind_nodes[i]).collect();
            nn.sort();
            nn.dedup();
            let mut ss = 0.0f32;
            for a in 0..nn.len() {
                for b in a + 1..nn.len() {
                    let r = Vec3::from_array(v.node_rest[nn[a] as usize]).distance(Vec3::from_array(v.node_rest[nn[b] as usize]));
                    ss = ss.max(nodes[nn[a] as usize].distance(nodes[nn[b] as usize]) / r.max(1e-4));
                }
            }
            sk_st.push(ss);
            eprintln!("  shard {} stretch {:.1} drawn {:.2} m; skeleton stretch over its nodes {:.2}", piece(ids[0]), worst, wl, ss);
            for &i in ids {
                let bt = v.bind_nodes[i];
                let tri_now = [nodes[bt[0] as usize], nodes[bt[1] as usize], nodes[bt[2] as usize]];
                let tri_rest = bt.map(|k| Vec3::from_array(v.node_rest[k as usize]));
                let nn = (tri_now[1] - tri_now[0]).cross(tri_now[2] - tri_now[0]).normalize_or_zero();
                let nr = (tri_rest[1] - tri_rest[0]).cross(tri_rest[2] - tri_rest[0]).normalize_or_zero();
                eprintln!(
                    "    v{i} rigid {} smooth {} alt {} part {} nodes {:?} abh {:.2?} rest |e| {:.3?} now |e| [{:.3}, {:.3}] tri normal turned {:.0}° moved {:.3} m tear {:.2}",
                    v.rigid[i], v.smooth[i], v.alt.get(i).copied().unwrap_or(0), v.part[i], bt, v.bind_coords[i], v.bind_rest[i],
                    tri_now[1].distance(tri_now[0]), tri_now[2].distance(tri_now[0]),
                    nn.dot(nr).clamp(-1.0, 1.0).acos().to_degrees(), pos[i].distance(Vec3::from_array(v.positions[i])), vt[i]
                );
            }
        }
    }
    if let Some(path) = obj {
        let mut head = String::new();
        for q in &pos {
            let q = if q.x.is_finite() { *q } else { Vec3::ZERO };
            head.push_str(&format!("v {} {} {}\n", q.x, q.y, q.z));
        }
        std::fs::write(path, head + &out).unwrap();
        let stem = path.trim_end_matches(".obj");
        for (k, dir) in [Vec3::new(0.8, 0.45, 1.0), Vec3::new(-1.0, 0.35, -0.2)].iter().enumerate() {
            render_png(v, &pos, &vt, &format!("{stem}_view{k}.png"), *dir);
        }
    }
    st
}

/// The camera: looking at `c` from along `dir` (toward the eye), `rad`
/// half the view's height (orthographic).
#[derive(Clone, Copy)]
struct Cam {
    c: Vec3,
    rad: f32,
    dir: Vec3,
}

/// A camera framing `pos` (1st–99th percentile) from `dir`.
fn frame_cam(pos: &[Vec3], dir: Vec3) -> Cam {
    let fin: Vec<Vec3> = pos.iter().copied().filter(|p| p.is_finite()).collect();
    let pct = |f: fn(&Vec3) -> f32, q: f32| {
        let mut a: Vec<f32> = fin.iter().map(f).collect();
        a.sort_by(f32::total_cmp);
        a[((a.len() - 1) as f32 * q) as usize]
    };
    let lo = Vec3::new(pct(|p| p.x, 0.01), pct(|p| p.y, 0.01), pct(|p| p.z, 0.01));
    let hi = Vec3::new(pct(|p| p.x, 0.99), pct(|p| p.y, 0.99), pct(|p| p.z, 0.99));
    Cam {
        c: 0.5 * (lo + hi),
        rad: 0.5 * (hi - lo).length() * 1.25,
        dir,
    }
}

/// A CPU z-buffer render of the drawn skin (the renderer's cut: per pixel,
/// the tear interpolated across the triangle, discarded past ½), lit flat
/// from its face normal; back faces in dark red so flips show. With `lines`
/// (node pairs and colours) the skeleton is drawn over it.
fn raster(
    v: &ShowcaseVisual,
    pos: &[Vec3],
    tear: &[f32],
    cam: Cam,
    (w, h): (usize, usize),
    lines: Option<(&[Vec3], &[(u32, u32, [u8; 3])])>,
) -> Vec<[u8; 3]> {
    let fwd = -cam.dir.normalize();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    let scale = (h as f32 * 0.5) / cam.rad;
    let proj = |p: Vec3| {
        let d = p - cam.c;
        (w as f32 * 0.5 + d.dot(right) * scale, h as f32 * 0.5 - d.dot(up) * scale, d.dot(fwd))
    };
    let dir = cam.dir.normalize();
    let mut z = vec![f32::MAX; w * h];
    let mut img = vec![[38u8, 42, 52]; w * h];
    let light = Vec3::new(0.4, 0.8, 0.45).normalize();
    let ghost = lines.is_some();
    for t in v.indices.chunks_exact(3) {
        let ids = [t[0] as usize, t[1] as usize, t[2] as usize];
        let p = ids.map(|i| pos[i]);
        if p.iter().any(|q| !q.is_finite()) {
            continue;
        }
        let tv = ids.map(|i| tear[i]);
        if tv.iter().all(|&x| x >= 0.5) {
            continue;
        }
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        let vn: Vec3 = ids.iter().map(|&i| Vec3::from_array(v.normals[i])).sum();
        let r0 = ids.map(|i| Vec3::from_array(v.positions[i]));
        let outward = (r0[1] - r0[0]).cross(r0[2] - r0[0]).dot(vn) >= 0.0;
        let nn = if outward { n } else { -n };
        let facing = nn.dot(dir) >= 0.0;
        let col = v.colors[ids[0]];
        let lit = 0.25 + 0.75 * nn.dot(light).max(0.0);
        let dim = if ghost { 0.45 } else { 1.0 };
        let rgb = if facing {
            [col[0], col[1], col[2]].map(|x| ((x.powf(1.0 / 2.2) * lit * dim).min(1.0) * 255.0) as u8)
        } else {
            [(110.0 * lit * dim) as u8, 10, 10]
        };
        let s = p.map(proj);
        let (x0, x1) = (s.iter().map(|q| q.0).fold(f32::MAX, f32::min).max(0.0), s.iter().map(|q| q.0).fold(f32::MIN, f32::max).min(w as f32 - 1.0));
        let (y0, y1) = (s.iter().map(|q| q.1).fold(f32::MAX, f32::min).max(0.0), s.iter().map(|q| q.1).fold(f32::MIN, f32::max).min(h as f32 - 1.0));
        let area = (s[1].0 - s[0].0) * (s[2].1 - s[0].1) - (s[2].0 - s[0].0) * (s[1].1 - s[0].1);
        if area.abs() < 1e-9 || x0 > x1 || y0 > y1 {
            continue;
        }
        for y in y0 as usize..=y1 as usize {
            for x in x0 as usize..=x1 as usize {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((s[1].0 - px) * (s[2].1 - py) - (s[2].0 - px) * (s[1].1 - py)) / area;
                let w1 = ((s[2].0 - px) * (s[0].1 - py) - (s[0].0 - px) * (s[2].1 - py)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                if w0 * tv[0] + w1 * tv[1] + w2 * tv[2] >= 0.5 {
                    continue;
                }
                let d = w0 * s[0].2 + w1 * s[1].2 + w2 * s[2].2;
                let k = y * w + x;
                if d < z[k] {
                    z[k] = d;
                    img[k] = rgb;
                }
            }
        }
    }
    if let Some((nodes, beams)) = lines {
        for &(a, b, col) in beams {
            let (pa, pb) = (proj(nodes[a as usize]), proj(nodes[b as usize]));
            let steps = ((pb.0 - pa.0).abs().max((pb.1 - pa.1).abs()) as usize).clamp(1, 4000);
            for k in 0..=steps {
                let t = k as f32 / steps as f32;
                let (x, y) = (pa.0 + (pb.0 - pa.0) * t, pa.1 + (pb.1 - pa.1) * t);
                if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
                    img[y as usize * w + x as usize] = col;
                }
            }
        }
    }
    img
}

fn write_png(img: &[[u8; 3]], (w, h): (usize, usize), path: &str) {
    let f = std::fs::File::create(path).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    let mut wr = e.write_header().unwrap();
    wr.write_image_data(&img.concat()).unwrap();
}

fn render_png(v: &ShowcaseVisual, pos: &[Vec3], tear: &[f32], path: &str, dir: Vec3) {
    let size = (1400, 900);
    let img = raster(v, pos, tear, frame_cam(pos, dir), size, None);
    write_png(&img, size, path);
}

/// The skin of `v` for solver `s`: positions and tears as drawn.
fn drawn_skin(v: &ShowcaseVisual, s: &ExplicitBeams) -> (Vec<Vec3>, Vec<f32>) {
    let nodes = s.positions();
    let bs = states(s);
    let us = unit_states(v, &bs);
    let (pos, _) = deform_beams(v, &nodes, Some(&us), Some(&bs));
    let t = vertex_tears(v, &nodes, Some(&bs), &pos);
    (pos, t)
}

/// The skeleton's live beams (green; yellow → red by plastic strain).
fn skeleton(s: &ExplicitBeams) -> Vec<(u32, u32, [u8; 3])> {
    s.beams
        .iter()
        .filter(|b| !b.broken && b.k > 0.0)
        .map(|b| {
            let pl = ((b.rest - b.rest0).abs() / b.rest0.max(1e-6) / 0.08).min(1.0);
            let c = if pl < 0.0125 { [40, 220, 90] } else { [255, (215.0 * (1.0 - pl)) as u8 + 25, 20] };
            (b.a, b.b, c)
        })
        .collect()
}

/// A side-by-side film frame: the triangle binding (left) and the node-frame
/// skin (right) of the same solver state, same camera.
struct Film {
    dir: String,
    k: usize,
    old: ShowcaseVisual,
    new: ShowcaseVisual,
}

impl Film {
    fn frame(&mut self, s: &ExplicitBeams, cam: Cam, overlay: bool) {
        let path = format!("{}/f{:05}.png", self.dir, self.k);
        self.k += 1;
        if std::path::Path::new(&path).exists() {
            return;
        }
        let (w, h) = (960usize, 720usize);
        let nodes = s.positions();
        let sk = skeleton(s);
        let lines = overlay.then_some((&nodes[..], &sk[..]));
        let (po, to) = drawn_skin(&self.old, s);
        let (pn, tn) = drawn_skin(&self.new, s);
        let a = raster(&self.old, &po, &to, cam, (w, h), lines);
        let b = raster(&self.new, &pn, &tn, cam, (w, h), lines);
        let mut img = Vec::with_capacity(2 * w * h);
        for y in 0..h {
            img.extend_from_slice(&a[y * w..(y + 1) * w]);
            img.extend_from_slice(&b[y * w..(y + 1) * w]);
        }
        write_png(&img, (2 * w, h), &path);
    }

    /// An orbit of the settled state: `n` frames round the vertical, skin
    /// then with the skeleton.
    fn orbit(&mut self, s: &ExplicitBeams, cam: Cam, n: usize) {
        for overlay in [false, true] {
            for k in 0..n {
                let a = std::f32::consts::TAU * k as f32 / n as f32;
                let d = Quat::from_rotation_y(a) * cam.dir;
                self.frame(s, Cam { dir: d, ..cam }, overlay);
            }
        }
    }
}

fn film(b: &Built, scene: &str, out: &str) {
    let mut old = showcase_visual_triangles_lod(b, 1.0).unwrap();
    raycast_visual(&mut old, &b.truck).unwrap();
    let new = visual_frames(b);
    let dir = format!("{out}/{scene}");
    std::fs::create_dir_all(&dir).unwrap();
    let mut f = Film { dir, k: 0, old, new };
    match scene {
        "wall" => {
            let vel = Vec3::new(0.0, 0.0, 70.0 / 3.6);
            let mut s = spawn(b, Vec3::ZERO, 0.0, vel);
            let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
            s.boxes.push(StaticBox::centered(Vec3::new(0.0, 1.5, front + 0.6), Vec3::new(8.0, 1.5, 0.3)));
            // Close on the grille and bonnet: the front, from ahead-left, up.
            let cam = Cam { c: Vec3::new(0.0, 0.6, front - 0.3), rad: 1.5, dir: Vec3::new(-0.7, 0.55, 1.0).normalize() };
            for k in 0..90 {
                s.frame();
                if k < 45 || k % 3 == 0 {
                    f.frame(&s, cam, false);
                }
            }
            f.orbit(&s, Cam { c: cam.c - Vec3::Z * 1.0, rad: 2.2, ..cam }, 48);
        }
        "side" => {
            let struck = spawn(b, Vec3::ZERO, 0.0, Vec3::ZERO);
            let p0 = struck.positions();
            let rear = b.skeleton.parts.iter().position(|p| p == "door_r_l").unwrap() as u16;
            let door: Vec<Vec3> = (0..b.skeleton.nodes.len()).filter(|&i| b.skeleton.nodes[i].part == rear).map(|i| p0[i]).collect();
            let zd = door.iter().map(|p| p.z).sum::<f32>() / door.len() as f32;
            let hw = p0.iter().map(|p| p.x).fold(f32::MIN, f32::max);
            let len = b.def.profile.length;
            let striker = spawn(b, Vec3::new(hw + 0.5 * len + 0.3, 0.0, zd), -90.0, Vec3::new(-60.0 / 3.6, 0.0, 0.0));
            let mut w = ExplicitBeamsWorld::new(vec![struck, striker]);
            // The struck car's rear door, from its left, the striker out of view behind the camera's side.
            let cam = Cam { c: Vec3::new(hw, 0.7, zd), rad: 1.6, dir: Vec3::new(1.0, 0.45, -0.35).normalize() };
            for k in 0..90 {
                w.frame_60();
                if k < 45 || k % 3 == 0 {
                    f.frame(&w.vehicles[0], cam, false);
                }
            }
            f.orbit(&w.vehicles[0], Cam { c: Vec3::new(0.0, 0.7, 0.0) + centroid(&w.vehicles[0].positions()) * Vec3::new(1.0, 0.0, 1.0), rad: 3.0, ..cam }, 48);
        }
        _ => {
            // The grab: the front left door's outermost node 0.5, 2 and 10 m
            // outward (50 kN/m, the page's default), each released.
            use engine_softbody::explicit_beams::{Grab, STEP};
            let mut s = spawn(b, Vec3::ZERO, 0.0, Vec3::ZERO);
            s.run(0.3);
            let part = scene.strip_prefix("grab_").unwrap_or("door_f_l");
            let pi = b.skeleton.parts.iter().position(|p| p == part).unwrap() as u16;
            let p = s.positions();
            let c = centroid(&p);
            let node = (0..b.skeleton.nodes.len())
                .filter(|&i| b.skeleton.nodes[i].part == pi)
                .max_by(|&x, &y| (p[x] - c).length().total_cmp(&(p[y] - c).length()))
                .unwrap();
            let dirn = ((p[node] - c) * Vec3::new(1.0, 0.3, 1.0)).normalize();
            let k = 5.0e4;
            let m = s.nodes[node].mass;
            let cd = (2.0 * (k * m).sqrt()).min(0.5 * m / STEP);
            let cam = Cam { c: c + dirn * 1.5, rad: 3.2, dir: (dirn + Vec3::new(0.0, 0.6, 0.0) + dirn.cross(Vec3::Y) * 0.6).normalize() };
            for dist in [0.5f32, 2.0, 10.0] {
                let start = s.positions()[node];
                for k2 in 0..120u32 {
                    s.grab = (k2 < 60).then(|| Grab { node: node as u32, target: start + dirn * dist * (k2 as f32 / 30.0).min(1.0), k, c: cd });
                    s.frame();
                    if k2 % 2 == 0 {
                        f.frame(&s, Cam { rad: if dist > 5.0 { 7.0 } else { cam.rad }, c: if dist > 5.0 { c + dirn * 5.0 } else { cam.c }, ..cam }, false);
                    }
                }
            }
            s.grab = None;
            f.orbit(&s, Cam { c, rad: 3.0, ..cam }, 48);
        }
    }
    eprintln!("{scene}: {} frames in {out}/{scene}", f.k);
}

struct Car {
    rest_centre: Vec3,
    radius: f32,
}

fn car_of(s: &ExplicitBeams) -> Car {
    let p = s.positions();
    let c = centroid(&p);
    let radius = p.iter().map(|q| q.distance(c)).fold(0.0, f32::max);
    Car {
        rest_centre: c,
        radius,
    }
}

/// The median position per axis (robust to a few flung nodes).
fn centroid(p: &[Vec3]) -> Vec3 {
    let med = |f: fn(&Vec3) -> f32| {
        let mut v: Vec<f32> = p.iter().map(f).collect();
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    Vec3::new(med(|q| q.x), med(|q| q.y), med(|q| q.z))
}

fn flung(s: &ExplicitBeams, car: &Car, margin: f32) -> Vec<bool> {
    let p = s.positions();
    let c = centroid(&p);
    p.iter().map(|q| q.distance(c) > car.radius + margin).collect()
}

fn describe_node(s: &ExplicitBeams, b: &Built, i: usize) {
    let p = s.positions();
    let part = b
        .skeleton
        .nodes
        .get(i)
        .map(|n| b.skeleton.parts.get(n.part as usize).cloned().unwrap_or_default())
        .unwrap_or_else(|| "(solver-only)".into());
    let n = &s.nodes[i];
    let live: Vec<u32> = s.incidence[i]
        .iter()
        .copied()
        .filter(|&j| !s.beams[j as usize].broken)
        .collect();
    eprintln!(
        "    node {i} part {part} mass {:.3} kg at {:.2?} speed {:.1} m/s; beams {} live {}",
        n.mass,
        p[i],
        n.vel.length(),
        s.incidence[i].len(),
        live.len()
    );
    for &j in live.iter().take(12) {
        let bm = &s.beams[j as usize];
        let o = if bm.a as usize == i { bm.b } else { bm.a } as usize;
        let len = p[i].distance(p[o]);
        eprintln!(
            "      live beam {j} kind {:?} k {:.3e} rest0 {:.3} rest {:.3} len {:.3} strength {:.3e} k|dL|/S {:.2} → node {o}",
            bm.kind,
            bm.k,
            bm.rest0,
            bm.rest,
            len,
            bm.strength,
            bm.k * (len - bm.rest).abs() / bm.strength.max(1.0)
        );
    }
}

fn report(tag: &str, v: &ShowcaseVisual, b: &Built, s: &ExplicitBeams, car: &Car, obj: Option<String>) {
    let fl = flung(s, car, 0.5);
    let nf = fl.iter().filter(|&&x| x).count();
    let (br, bent) = s.damage();
    let st = skin_stats(v, s, &fl, obj.as_deref());
    eprintln!(
        "{tag}: broken {br} bent {bent} / {} beams; flung nodes {nf}; drawn {} / {} tris; shards {} (flung {} split {} lattice {} of which seam {}) longest {:.2} m; stretch max {:.1} p99 {:.2}; flipped {}; split vertices {}; jumpy vertices {}",
        s.beams.len(),
        st.drawn,
        st.tris,
        st.shards,
        st.by_flung,
        st.by_split,
        st.by_lattice,
        st.lattice_seam,
        st.long_m,
        st.max_stretch,
        st.p99_stretch,
        st.flipped,
        st.split_vertices,
        st.jumpy
    );
    let _ = car.rest_centre;
    for (i, _) in fl.iter().enumerate().filter(|(_, &x)| x).take(6) {
        describe_node(s, b, i);
    }
}

fn visual_frames(b: &Built) -> ShowcaseVisual {
    let mut v = showcase_visual_lod(b, 1.0).unwrap();
    raycast_visual(&mut v, &b.truck).unwrap();
    v
}

fn visual(b: &Built) -> ShowcaseVisual {
    // BIND=triangles: the binding before the node-frame skin (A/B).
    let mut v = if std::env::var("BIND").is_ok_and(|x| x == "triangles") {
        showcase_visual_triangles_lod(b, 1.0).unwrap()
    } else {
        showcase_visual_lod(b, 1.0).unwrap()
    };
    raycast_visual(&mut v, &b.truck).unwrap();
    v
}

/// Node speed histories: the fastest node per frame, and every node's peak.
struct Track {
    peak: Vec<(f32, u32)>,
}

impl Track {
    fn new(n: usize) -> Self {
        Self {
            peak: vec![(0.0, 0); n],
        }
    }
    fn see(&mut self, s: &ExplicitBeams, frame: u32) {
        for (i, n) in s.nodes.iter().enumerate() {
            let sp = n.vel.length();
            if sp > self.peak[i].0 {
                self.peak[i] = (sp, frame);
            }
        }
    }
}

fn mech(s: &ExplicitBeams) -> f64 {
    let mut e = engine_softbody::explicit_contact::elastic_energy(s);
    for n in &s.nodes {
        e += 0.5 * n.mass as f64 * n.vel.length_squared() as f64
            + (n.mass * 9.81) as f64 * (s.origin.y + n.pos.y) as f64;
    }
    e
}

fn wall(b: &Built, v: &ShowcaseVisual, kmh: f32, hits: u32, out: Option<&str>) {
    let vel = Vec3::new(0.0, 0.0, kmh / 3.6);
    let mut s = spawn(b, Vec3::ZERO, 0.0, vel);
    let car = car_of(&s);
    let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    let wall = StaticBox::centered(Vec3::new(0.0, 1.5, front + 0.3 + 0.3), Vec3::new(8.0, 1.5, 0.3));
    s.boxes.push(wall);
    let mut tr = Track::new(s.nodes.len());
    let tag = format!("wall {kmh} km/h");
    report(&format!("{tag} rest"), v, b, &s, &car, out.map(|d| format!("{d}/wall{kmh}_rest.obj")));
    let mut frame = 0u32;
    let mut e_prev = mech(&s);
    let mut rises = Vec::new();
    for hit in 0..hits {
        if hit > 0 {
            // Back off and hit again: the crashed car given the speed again,
            // from 1 m off the wall.
            let back = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max) - wall.lo.z + 1.0;
            for n in &mut s.nodes {
                n.pos.z -= back;
                n.vel = vel;
            }
            e_prev = mech(&s);
        }
        for f in 0..90 {
            s.frame();
            frame += 1;
            tr.see(&s, frame);
            let e = mech(&s);
            if e > e_prev + 50.0 {
                rises.push((frame, e - e_prev));
            }
            e_prev = e;
            if f % 15 == 14 {
                report(&format!("{tag} hit {} t {:.2}", hit + 1, (f + 1) as f32 / 60.0), v, b, &s, &car, None);
            }
        }
    }
    report(&format!("{tag} end"), v, b, &s, &car, out.map(|d| format!("{d}/wall{kmh}_x{hits}.obj")));
    eprintln!("  energy rises > 50 J per frame: {} {:?}", rises.len(), &rises[..rises.len().min(8)]);
    let fl = flung(&s, &car, 0.5);
    for (i, _) in fl.iter().enumerate().filter(|(_, &x)| x).take(6) {
        eprintln!("  flung node {i} peak speed {:.1} m/s at frame {}", tr.peak[i].0, tr.peak[i].1);
    }
}

fn side(b: &Built, v: &ShowcaseVisual, kmh: f32, out: Option<&str>) {
    // The struck car parked at the origin facing +z; the striker comes in
    // along −x at the rear door.
    let struck = spawn(b, Vec3::ZERO, 0.0, Vec3::ZERO);
    let car = car_of(&struck);
    let p0 = struck.positions();
    let rear_part = b.skeleton.parts.iter().position(|p| p == "door_r_l").unwrap() as u16;
    let door: Vec<Vec3> = (0..b.skeleton.nodes.len())
        .filter(|&i| b.skeleton.nodes[i].part == rear_part)
        .map(|i| p0[i])
        .collect();
    let zd = door.iter().map(|p| p.z).sum::<f32>() / door.len() as f32;
    let hw = p0.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let len = b.def.profile.length;
    let striker = spawn(
        b,
        Vec3::new(hw + 0.5 * len + 0.3, 0.0, zd),
        -90.0,
        Vec3::new(-kmh / 3.6, 0.0, 0.0),
    );
    let car2 = car_of(&striker);
    let mut w = ExplicitBeamsWorld::new(vec![struck, striker]);
    let tag = format!("side {kmh} km/h");
    let mut tr = Track::new(w.vehicles[0].nodes.len());
    for f in 0..120u32 {
        w.frame_60();
        tr.see(&w.vehicles[0], f);
        if f % 20 == 19 {
            report(&format!("{tag} struck t {:.2}", (f + 1) as f32 / 60.0), v, b, &w.vehicles[0], &car, None);
            report(&format!("{tag} striker t {:.2}", (f + 1) as f32 / 60.0), v, b, &w.vehicles[1], &car2, None);
        }
    }
    report(&format!("{tag} struck end"), v, b, &w.vehicles[0], &car, out.map(|d| format!("{d}/side{kmh}_struck.obj")));
    report(&format!("{tag} striker end"), v, b, &w.vehicles[1], &car2, out.map(|d| format!("{d}/side{kmh}_striker.obj")));
}

/// The page's grab tool: a node on `part` (the one nearest the part's
/// outermost point) pulled outward `dist` m at stiffness `k` over 0.5 s
/// (the user dragging), held 1 s, released, then 1 s more.
fn grab(b: &Built, v: &ShowcaseVisual, part: &str, dist: f32, k: f32, out: Option<&str>) {
    use engine_softbody::explicit_beams::{Grab, STEP};
    let mut s = spawn(b, Vec3::ZERO, 0.0, Vec3::ZERO);
    s.run(0.5);
    let car = car_of(&s);
    let pi = b.skeleton.parts.iter().position(|p| p == part).unwrap() as u16;
    let p = s.positions();
    let c = centroid(&p);
    let node = (0..b.skeleton.nodes.len())
        .filter(|&i| b.skeleton.nodes[i].part == pi)
        .max_by(|&x, &y| (p[x] - c).length().total_cmp(&(p[y] - c).length()))
        .unwrap();
    let dir = ((p[node] - c) * Vec3::new(1.0, 0.3, 1.0)).normalize();
    let start = p[node];
    let m = s.nodes[node].mass;
    let cdamp = (2.0 * (k * m).sqrt()).min(0.5 * m / STEP);
    let tag = format!("grab {part} node {node} {dist} m k {k:.0e}");
    let mut tr = Track::new(s.nodes.len());
    let mut e_prev = mech(&s);
    let mut rises = Vec::new();
    for f in 0..150u32 {
        let t = (f as f32 / 30.0).min(1.0);
        if f < 90 {
            s.grab = Some(Grab { node: node as u32, target: start + dir * dist * t, k, c: cdamp });
        } else {
            s.grab = None;
        }
        s.frame();
        tr.see(&s, f);
        let e = mech(&s);
        if f >= 90 && e > e_prev + 20.0 {
            rises.push((f, e - e_prev));
        }
        e_prev = e;
        if f == 89 {
            report(&format!("{tag} held"), v, b, &s, &car, out.map(|d| format!("{d}/grab_{part}_{dist}_{k:.0e}_held.obj")));
        }
    }
    report(&format!("{tag} released"), v, b, &s, &car, out.map(|d| format!("{d}/grab_{part}_{dist}_{k:.0e}_rel.obj")));
    eprintln!("  after release, energy rises > 20 J per frame: {} {:?}", rises.len(), &rises[..rises.len().min(5)]);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let scene = a.get(1).map(String::as_str).unwrap_or("all");
    let kmh: f32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(70.0);
    let b = build(
        &style("sedan").unwrap(),
        &BuildParams {
            resolution: Resolution::Medium,
            ..Default::default()
        },
    )
    .unwrap();
    let v = visual(&b);
    let s = spawn(&b, Vec3::ZERO, 0.0, Vec3::ZERO);
    eprintln!(
        "sedan medium raycast: {} nodes {} beams; visual {} vertices {} tris; weighted {}; beamng profile {} centred damping {}",
        s.nodes.len(),
        s.beams.len(),
        v.positions.len(),
        v.indices.len() / 3,
        v.wt_start.len() == v.positions.len() + 1,
        s.beamng_profile,
        s.settings.centred_damping
    );
    if scene == "zerok" {
        zero_k_census(&b, &s);
        return;
    }
    let out = std::env::var("OBJ").ok();
    if let Some(d) = &out {
        std::fs::create_dir_all(d).unwrap();
    }
    let out = out.as_deref();
    if let Some(sc) = scene.strip_prefix("film:") {
        film(&b, sc, &std::env::var("FILM").unwrap_or_else(|_| "out/skin-film".into()));
        return;
    }
    match scene {
        "wall" => wall(&b, &v, kmh, 1, out),
        "wall2" => wall(&b, &v, kmh, 3, out),
        "side" => side(&b, &v, kmh, out),
        "grab" => {
            let parts: Vec<String> = std::env::var("PARTS").map(|p| p.split(',').map(String::from).collect()).unwrap_or_else(|_| {
                ["door_f_l", "fender_l", "bonnet", "bumper_f", "door_r_r", "bootlid"].map(String::from).to_vec()
            });
            for part in &parts {
                for d in [0.5, 2.0, 10.0] {
                    for k in [5e4, 3e6] {
                        grab(&b, &v, part, d, k, out);
                    }
                }
            }
        }
        _ => {
            for k in [50.0, 90.0] {
                wall(&b, &v, k, 1, out);
            }
            wall(&b, &v, 60.0, 3, out);
            side(&b, &v, 60.0, out);
        }
    }
}

/// Beams with no stiffness (they never strain, so the lab draws them
/// low-strain at any length), by kind and parts.
#[allow(dead_code)]
pub fn zero_k_census(b: &Built, s: &ExplicitBeams) {
    let mut m: std::collections::BTreeMap<String, usize> = Default::default();
    let part = |i: u32| {
        b.skeleton
            .nodes
            .get(i as usize)
            .map(|n| b.skeleton.parts[n.part as usize].clone())
            .unwrap_or_else(|| "-".into())
    };
    for bm in s.beams.iter().filter(|bm| bm.k == 0.0) {
        *m.entry(format!("{:?} {:?} {}–{} damp {}", bm.kind, bm.ty, part(bm.a), part(bm.b), bm.c > 0.0)).or_default() += 1;
    }
    eprintln!("zero-stiffness beams: {}", m.values().sum::<usize>());
    for (k, n) in m {
        eprintln!("  {n:5} {k}");
    }
}

