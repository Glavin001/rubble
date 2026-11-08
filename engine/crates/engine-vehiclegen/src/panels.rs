//! Turning welded skin patches into deliverable panels: shut-line insets,
//! rolled hems, inner panels, and glass made from the openings.
//!
//! A line-for-line port of `vehgen/panels.py`. Where Python's iteration order
//! decides the result (dict insertion order in [`boundary_loops`]) it is
//! reproduced exactly; Python `set` iteration (neighbour lists) only ever
//! feeds means, so [`neighbours`] returns sorted lists and results agree with
//! Python up to floating-point summation order.

use std::collections::{HashMap, HashSet};

use glam::{DVec2, DVec3};

use crate::mesh::{key3, Mat, Mesh};

/// how far the roll's apex reaches past the skin edge
pub const HEM_OUT: f64 = 0.0025;
/// depth of the apex below the skin
pub const HEM_DROP: f64 = 0.0040;
/// flange returns this far under the skin
pub const FLANGE_IN: f64 = 0.0070;
pub const FLANGE_DROP: f64 = 0.0085;

/// A position key: `mesh::key3(p, 1e-6)` (the Python `np.round(v / 1e-6)`).
pub type Key = (i64, i64, i64);

#[inline]
fn nrm(x: DVec3) -> DVec3 {
    x / x.length().max(1e-12)
}

/// Python's float floor division `a // b` (CPython `float_floor_div`).
fn py_floordiv(vx: f64, wx: f64) -> f64 {
    let m = vx % wx; // C fmod
    let mut div = (vx - m) / wx;
    if m != 0.0 && ((wx < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let mut fd = div.floor();
        if div - fd > 0.5 {
            fd += 1.0;
        }
        fd
    } else {
        0.0f64.copysign(vx / wx)
    }
}

/// `np.percentile(a, p)` with the default 'linear' method (numpy 1.24).
fn percentile_linear(a: &[f64], p: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = s.len();
    let q = p / 100.0;
    let vi = (n as f64 - 1.0) * q;
    let prev = vi.floor();
    let pi = (prev as usize).min(n - 1);
    let ni = (pi + 1).min(n - 1);
    let t = vi - prev;
    let (a0, b0) = (s[pi], s[ni]);
    let diff = b0 - a0;
    if t >= 0.5 {
        b0 - diff * (1.0 - t)
    } else {
        a0 + diff * t
    }
}

/// Mean of points, summed in order then divided (numpy `mean(0)`).
fn mean_of(v: &[DVec3], ix: &[u32]) -> DVec3 {
    let mut s = DVec3::ZERO;
    for &i in ix {
        s += v[i as usize];
    }
    s / ix.len() as f64
}

/// Directed boundary edges (face on the left seen from the front) chained
/// into loops.
pub fn boundary_loops(m: &Mesh) -> Vec<Vec<u32>> {
    // `cnt` as a Python dict: insertion-ordered edges plus a lookup
    let mut order: Vec<(u32, u32)> = Vec::new();
    let mut cnt: HashMap<(u32, u32), usize> = HashMap::new();
    for f in m.faces() {
        let k = f.len();
        for i in 0..k {
            let e = (f[i], f[(i + 1) % k]);
            let c = cnt.entry(e).or_insert(0);
            if *c == 0 {
                order.push(e);
            }
            *c += 1;
        }
    }
    let bnd: Vec<(u32, u32)> = order.iter().copied().filter(|&(a, b)| !cnt.contains_key(&(b, a))).collect();
    let mut nxt: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &bnd {
        nxt.entry(a).or_default().push(b);
    }
    let mut used: HashSet<(u32, u32)> = HashSet::new();
    let mut loops = Vec::new();
    for &(a, b) in &bnd {
        if used.contains(&(a, b)) {
            continue;
        }
        let mut lp = vec![a];
        used.insert((a, b));
        let mut cur = b;
        let mut guard = 0;
        while cur != a && guard < 200000 {
            lp.push(cur);
            let x = nxt.get(&cur).and_then(|c| c.iter().copied().find(|&x| !used.contains(&(cur, x))));
            let Some(x) = x else { break };
            used.insert((cur, x));
            cur = x;
            guard += 1;
        }
        loops.push(lp);
    }
    loops
}

/// Edge neighbours of every vertex (Python sets; here sorted lists).
pub fn neighbours(m: &Mesh) -> Vec<Vec<u32>> {
    let mut nb: Vec<Vec<u32>> = vec![Vec::new(); m.v.len()];
    for f in m.faces() {
        let k = f.len();
        for i in 0..k {
            let (a, b) = (f[i], f[(i + 1) % k]);
            nb[a as usize].push(b);
            nb[b as usize].push(a);
        }
    }
    for s in &mut nb {
        s.sort_unstable();
        s.dedup();
    }
    nb
}

/// Inward in-surface direction per loop vertex (miter-scaled, <= 1.6),
/// optionally smoothed along the loop (Python default `smooth=2`).
pub fn loop_dirs(m: &Mesh, lp: &[u32], n: &[DVec3], smooth: usize) -> Vec<DVec3> {
    let l = lp.len();
    let p: Vec<DVec3> = lp.iter().map(|&i| m.v[i as usize]).collect();
    let e: Vec<DVec3> = (0..l).map(|i| nrm(p[(i + 1) % l] - p[i])).collect();
    let nin: Vec<DVec3> = (0..l).map(|i| nrm(n[lp[i] as usize].cross(e[i]))).collect();
    let mut w: Vec<DVec3> = (0..l).map(|i| nrm(nin[i] + nin[(i + l - 1) % l])).collect();
    let cosh: Vec<f64> = (0..l).map(|i| w[i].dot(nin[i]).clamp(0.62, 1.0)).collect();
    for _ in 0..smooth {
        w = (0..l)
            .map(|i| nrm(0.25 * w[(i + l - 1) % l] + 0.5 * w[i] + 0.25 * w[(i + 1) % l]))
            .collect();
    }
    (0..l).map(|i| w[i] / cosh[i]).collect()
}

/// Move every boundary loop by d into the panel (negative d grows it),
/// spreading the move over several rings so thin rows keep their shape.
/// `d`: per-vertex distances (len = m.v.len()); `falloff` None = the
/// adaptive K-ring falloff.
pub fn inset_boundary(
    m: &mut Mesh,
    d: &[f64],
    falloff: Option<&[f64]>,
    loops: Option<Vec<Vec<u32>>>,
) -> (Vec<Vec<u32>>, Vec<bool>) {
    let loops = loops.unwrap_or_else(|| boundary_loops(m));
    let nv = m.v.len();
    assert_eq!(d.len(), nv, "inset_boundary: d must have one entry per vertex");
    let n = m.vertex_normals();
    let dv = d;
    let mut w = vec![DVec3::ZERO; nv];
    let mut isb = vec![false; nv];
    for lp in &loops {
        if lp.len() < 3 {
            continue;
        }
        let ld = loop_dirs(m, lp, &n, 2);
        for (k, &i) in lp.iter().enumerate() {
            w[i as usize] = ld[k];
            isb[i as usize] = true;
        }
    }
    let nb = neighbours(m);
    let auto: Vec<f64>;
    let falloff: &[f64] = match falloff {
        Some(f) => f,
        None => {
            // spread the move over enough rings that no row loses over 35 %
            // of its height: linear falloff over K rings
            let mut h1 = Vec::new();
            for vi in 0..nv {
                if !isb[vi] {
                    continue;
                }
                for &u in &nb[vi] {
                    if !isb[u as usize] {
                        h1.push((m.v[u as usize] - m.v[vi]).length());
                    }
                }
            }
            let hmin = if h1.is_empty() { 0.02 } else { percentile_linear(&h1, 1.0) };
            let dmax = dv.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x.abs()));
            let kf = (1.6 * dmax / (0.35 * hmin.max(1e-4))).ceil().clamp(2.0, 12.0);
            let k = kf as i64;
            auto = (1..k).map(|j| 1.0 - j as f64 / k as f64).collect();
            &auto
        }
    };
    let mut mv = vec![DVec3::ZERO; nv];
    for i in 0..nv {
        if isb[i] {
            mv[i] = w[i] * dv[i];
        }
    }
    let mut ring = isb.clone();
    let mut frontier: Vec<usize> = (0..nv).filter(|&i| isb[i]).collect();
    let mut src_w: Vec<DVec3> = (0..nv).map(|i| w[i] * dv[i]).collect();
    for &fac in falloff {
        let mut newf = Vec::new();
        // dict insertion order: keys in first-seen order
        let mut acc_order: Vec<usize> = Vec::new();
        let mut acc: HashMap<usize, Vec<DVec3>> = HashMap::new();
        for &vi in &frontier {
            for &u in &nb[vi] {
                let u = u as usize;
                if ring[u] {
                    continue;
                }
                acc.entry(u)
                    .or_insert_with(|| {
                        acc_order.push(u);
                        Vec::new()
                    })
                    .push(src_w[vi]);
            }
        }
        for &u in &acc_order {
            let ws = &acc[&u];
            let mut s = DVec3::ZERO;
            for x in ws {
                s += *x;
            }
            let wm = s / ws.len() as f64;
            src_w[u] = wm;
            mv[u] = wm * fac;
            ring[u] = true;
            newf.push(u);
        }
        frontier = newf;
    }
    for (p, x) in m.v.iter_mut().zip(&mv) {
        *p += *x;
    }
    (loops, isb)
}

/// Roll every border under the skin: apex row and flange row. Returns the
/// next free island id (Python default `island_base=1000`).
pub fn add_hem(m: &mut Mesh, loops: &[Vec<u32>], island_base: i32) -> i32 {
    let n = smooth_normals(m, 2);
    let mut hv: Vec<DVec3> = Vec::new();
    let mut hq: Vec<[u32; 4]> = Vec::new();
    let mut hquv: Vec<[DVec2; 4]> = Vec::new();
    let mut hqi: Vec<i32> = Vec::new();
    let mut island = island_base;
    let chunk = 1.2;
    let v1 = HEM_OUT.hypot(HEM_DROP);
    let v2 = v1 + (HEM_OUT + FLANGE_IN).hypot(FLANGE_DROP - HEM_DROP);
    for lp in loops {
        let l = lp.len();
        if l < 3 {
            continue;
        }
        let p: Vec<DVec3> = lp.iter().map(|&i| m.v[i as usize]).collect();
        let w = loop_dirs(m, lp, &n, 0);
        let seg: Vec<f64> = (0..l).map(|i| (p[(i + 1) % l] - p[i]).length()).collect();
        let base = (m.v.len() + hv.len()) as u32;
        let ia = |i: usize| base + i as u32;
        let ib = |i: usize| base + l as u32 + i as u32;
        let mut a_row = Vec::with_capacity(l);
        let mut b_row = Vec::with_capacity(l);
        for i in 0..l {
            let nn = n[lp[i] as usize];
            let local = seg[i].min(seg[(i + l - 1) % l]);
            let fin = FLANGE_IN.min(0.85 * local);
            a_row.push(p[i] - w[i] * HEM_OUT - nn * HEM_DROP);
            b_row.push(p[i] + w[i] * fin - nn * FLANGE_DROP);
        }
        hv.extend(a_row);
        hv.extend(b_row);
        let mut s = Vec::with_capacity(l + 1);
        s.push(0.0);
        let mut acc = 0.0;
        for &x in &seg {
            acc += x;
            s.push(acc);
        }
        for i in 0..l {
            let j = (i + 1) % l;
            let (a, b) = (lp[i], lp[j]);
            let (ua, ub) = (s[i], s[i + 1]);
            let cf = py_floordiv(ua, chunk);
            let c = cf as i32;
            let (ua0, ub0) = (ua - cf * chunk, ub - cf * chunk);
            hq.push([b, a, ia(i), ia(j)]);
            hquv.push([DVec2::new(ub0, 0.0), DVec2::new(ua0, 0.0), DVec2::new(ua0, v1), DVec2::new(ub0, v1)]);
            hqi.push(island + c);
            hq.push([ia(j), ia(i), ib(i), ib(j)]);
            hquv.push([DVec2::new(ub0, v1), DVec2::new(ua0, v1), DVec2::new(ua0, v2), DVec2::new(ub0, v2)]);
            hqi.push(island + c);
        }
        island += py_floordiv(s[l], chunk) as i32 + 1;
    }
    // the hem quads index the combined vertex array: append without offset
    let nh = hv.len();
    m.v.extend(hv);
    if let Some(col) = &mut m.col {
        col.extend(std::iter::repeat_n([1.0; 4], nh));
    }
    m.qm.extend(std::iter::repeat_n(Mat::Paint, hq.len()));
    m.q.extend(hq);
    m.quv.extend(hquv);
    m.qi.extend(hqi);
    island
}

/// Vertex normals averaged over the neighbourhood, so an offset copy does
/// not fold behind tight creases (Python default `it=6`).
pub fn smooth_normals(m: &Mesh, it: usize) -> Vec<DVec3> {
    let mut n = m.vertex_normals();
    let nb = neighbours(m);
    for _ in 0..it {
        let mut n2 = n.clone();
        for (i, ix) in nb.iter().enumerate() {
            if !ix.is_empty() {
                n2[i] = n[i] + mean_of(&n, ix);
            }
        }
        n = n2.into_iter().map(nrm).collect();
    }
    n
}

/// A simple inner panel: the skin's interior (`rings` rings in from the
/// border) pushed `depth` inward, facing the other way. Python defaults:
/// depth 0.035, island 2000, rings 2, mat paint.
pub fn inner_panel(skin: &Mesh, isb: &[bool], depth: f64, island: i32, rings: usize, mat: Mat) -> Mesh {
    let nb = neighbours(skin);
    let nv = skin.v.len();
    let mut ring = isb.to_vec();
    let mut frontier: Vec<usize> = (0..nv).filter(|&i| isb[i]).collect();
    for _ in 0..rings {
        let mut nf = Vec::new();
        for &vi in &frontier {
            for &u in &nb[vi] {
                let u = u as usize;
                if !ring[u] {
                    ring[u] = true;
                    nf.push(u);
                }
            }
        }
        frontier = nf;
    }
    let keep: Vec<bool> = ring.iter().map(|r| !r).collect();
    let n = smooth_normals(skin, 6);
    let mut remap = vec![u32::MAX; nv];
    let mut m = Mesh::new();
    for i in 0..nv {
        if keep[i] {
            remap[i] = m.v.len() as u32;
            m.v.push(skin.v[i] - n[i] * depth);
        }
    }
    let isl = |x: i32| if x >= 0 { island + x } else { -1 };
    let mut src: Vec<[u32; 4]> = Vec::new();
    for (k, f) in skin.q.iter().enumerate() {
        if f.iter().all(|&i| keep[i as usize]) {
            m.q.push(f.map(|i| remap[i as usize]));
            m.qm.push(mat);
            m.quv.push(skin.quv[k]);
            m.qi.push(isl(skin.qi[k]));
            src.push(*f);
        }
    }
    for (k, f) in skin.t.iter().enumerate() {
        if f.iter().all(|&i| keep[i as usize]) {
            m.t.push(f.map(|i| remap[i as usize]));
            m.tm.push(mat);
            m.tuv.push(skin.tuv[k]);
            m.ti.push(isl(skin.ti[k]));
        }
    }
    // drop faces the offset squeezed (behind tight creases): keep the panel
    // simple and well shaped rather than complete
    if !m.q.is_empty() {
        let edges = |p: [DVec3; 4]| -> [f64; 4] { std::array::from_fn(|j| (p[(j + 1) % 4] - p[j]).length()) };
        let fmin = |e: [f64; 4]| e.iter().copied().fold(f64::INFINITY, f64::min);
        let fmax = |e: [f64; 4]| e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ok: Vec<bool> = m
            .q
            .iter()
            .zip(&src)
            .map(|(f, sf)| {
                let p: [DVec3; 4] = f.map(|i| m.v[i as usize]);
                let d1 = (p[2] - p[0]).length();
                let d2 = (p[3] - p[1]).length();
                let e = edges(p);
                let mut ok = (fmin(e) > 0.25 * fmax(e)) & (d1.min(d2) > 0.5 * d1.max(d2) * 0.5);
                let es = edges(sf.map(|i| skin.v[i as usize]));
                ok &= fmin(e) > 0.5 * fmin(es);
                ok
            })
            .collect();
        crate::mesh::retain_by(&mut m.q, &ok);
        crate::mesh::retain_by(&mut m.qm, &ok);
        crate::mesh::retain_by(&mut m.quv, &ok);
        crate::mesh::retain_by(&mut m.qi, &ok);
    }
    m.flip();
    m
}

/// Tangential Laplacian smoothing of interior, non-crease vertices: evens out
/// sheared cells where blocks meet without moving the surface. Python
/// defaults: it 6, lam 0.35, crease_deg 12.
pub fn relax(m: &mut Mesh, it: usize, lam: f64, crease_deg: f64) {
    let nv = m.v.len();
    let loops = boundary_loops(m);
    let mut fixed = vec![false; nv];
    for lp in &loops {
        for &i in lp {
            fixed[i as usize] = true;
        }
    }
    // creases: vertices whose faces' normals spread more than crease_deg
    let faces = m.faces();
    let fnrm: Vec<DVec3> = faces
        .iter()
        .map(|f| {
            let p = |k: usize| m.v[f[k] as usize];
            let mut n = (p(1) - p(0)).cross(p(2) - p(0));
            if f.len() == 4 {
                n += (p(2) - p(0)).cross(p(3) - p(0));
            }
            n / n.length().max(1e-12)
        })
        .collect();
    let vn = m.vertex_normals();
    let mut mind = vec![1.0f64; nv];
    for (k, f) in faces.iter().enumerate() {
        for &i in f {
            let d = vn[i as usize].dot(fnrm[k]);
            mind[i as usize] = mind[i as usize].min(d);
        }
    }
    let cth = crease_deg.to_radians().cos();
    for i in 0..nv {
        fixed[i] |= mind[i] < cth;
    }
    let nb = neighbours(m);
    let free: Vec<usize> = (0..nv).filter(|&i| !fixed[i]).collect();
    for _ in 0..it {
        let vn = m.vertex_normals();
        let mut new = m.v.clone();
        for &i in &free {
            let ix = &nb[i];
            if ix.len() < 3 {
                continue;
            }
            let mut d = mean_of(&m.v, ix) - m.v[i];
            d -= vn[i] * d.dot(vn[i]);
            new[i] = m.v[i] + lam * d;
        }
        m.v = new;
    }
}

/// Pull sharp border corners in along their bisector: a radiused panel
/// corner, and hem rows that do not twist round it (Python default
/// `min_turn=35`).
pub fn round_corners(m: &mut Mesh, loops: &[Vec<u32>], min_turn: f64) {
    let n = m.vertex_normals();
    for lp in loops {
        let l = lp.len();
        if l < 5 {
            continue;
        }
        let p: Vec<DVec3> = lp.iter().map(|&i| m.v[i as usize]).collect();
        let e: Vec<DVec3> = (0..l).map(|i| p[(i + 1) % l] - p[i]).collect();
        let ln: Vec<f64> = e.iter().map(|x| x.length()).collect();
        let w = loop_dirs(m, lp, &n, 0);
        let mut upd = Vec::new();
        for i in 0..l {
            let ip = (i + l - 1) % l;
            let t0 = -e[ip] / ln[ip].max(1e-12);
            let t1 = e[i] / ln[i].max(1e-12);
            let turn = 180.0 - t0.dot(t1).clamp(-1.0, 1.0).acos().to_degrees();
            if turn > min_turn {
                let wl = nrm(w[i]);
                let amt = 0.35 * ln[i].min(ln[ip]) * ((turn - min_turn) / 55.0).clamp(0.0, 1.0);
                // numpy fancy `+=`: each read is the pre-update value, the
                // last write to a repeated index wins
                upd.push((lp[i] as usize, p[i] + wl * amt));
            }
        }
        for (i, x) in upd {
            m.v[i] = x;
        }
    }
}

/// block_meshes: list of (Mesh, island_id). Returns the finished Mesh.
/// Python defaults: gap 0.0046, hem true, inner None, island0 0, shut None.
pub fn panel(
    block_meshes: &[(Mesh, i32)],
    gap: f64,
    hem: bool,
    inner: Option<f64>,
    island0: i32,
    shut: Option<&HashSet<Key>>,
) -> Mesh {
    let mut m = Mesh::new();
    for (g, isl) in block_meshes {
        let mut g = g.clone();
        g.set_islands(island0 + isl);
        m.add(&g);
    }
    m.weld(1e-7);
    relax(&mut m, 6, 0.35, 12.0);
    let loops = boundary_loops(&m);
    let d0 = if hem { HEM_OUT + gap / 2.0 } else { gap / 2.0 };
    let d: Vec<f64> = match shut {
        // openings (wheel arches, glass, lamps) keep their drawn edge: only
        // shut lines against another panel are pulled back for the gap
        Some(shut) => m.v.iter().map(|p| if shut.contains(&key3(*p, 1e-6)) { d0 } else { HEM_OUT * 0.6 }).collect(),
        None => vec![d0; m.v.len()],
    };
    let (loops, isb) = inset_boundary(&mut m, &d, None, Some(loops));
    round_corners(&mut m, &loops, 35.0);
    let skin = m.clone();
    if hem {
        add_hem(&mut m, &loops, island0 + 100);
    }
    if let Some(depth) = inner {
        m.add(&inner_panel(&skin, &isb, depth, island0 + 1000, 3, Mat::Paint));
    }
    m
}

/// Glass from its opening: grown under the body flange, sunk `depth` below
/// the skin; edges shared with another pane are pulled back 3 mm. Python
/// defaults: depth 0.012, grow 0.006.
pub fn glass_from(block_meshes: &[(Mesh, i32)], depth: f64, grow: f64, shared: Option<&HashSet<Key>>) -> Mesh {
    let mut m = Mesh::new();
    for (g, _) in block_meshes {
        m.add(g);
    }
    m.weld(1e-7);
    let mut d = vec![-grow; m.v.len()];
    if let Some(shared) = shared.filter(|s| !s.is_empty()) {
        for (i, p) in m.v.iter().enumerate() {
            if shared.contains(&key3(*p, 1e-6)) {
                d[i] = 0.003;
            }
        }
    }
    inset_boundary(&mut m, &d, Some(&[0.5, 0.2]), None);
    let n = m.vertex_normals();
    for (p, nn) in m.v.iter_mut().zip(&n) {
        *p -= *nn * depth;
    }
    m.set_mat(Mat::Glass);
    m.set_islands(-1);
    m
}

/// A copy pushed `depth` back along the smoothed normals.
pub fn offset_copy(m: &Mesh, depth: f64, mat: Option<Mat>, flip: bool) -> Mesh {
    let mut o = m.clone();
    let n = smooth_normals(&o, 6);
    for (p, nn) in o.v.iter_mut().zip(&n) {
        *p -= *nn * depth;
    }
    if let Some(mat) = mat {
        o.set_mat(mat);
    }
    if flip {
        o.flip();
    }
    o.set_islands(-1);
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{grid, Grid2};

    /// The Python reference patch: points over i in [i0, i1), j in [0, 9),
    /// x = 0.1 i, z = 0.08 j, y = 0.3 + 0.15 sin(1.3 x) cos(0.9 z) + 0.05 x z,
    /// uv = (x, z).
    fn pts(i0: usize, i1: usize) -> (Grid2<DVec3>, Grid2<DVec2>) {
        let p = Grid2::from_fn(i1 - i0, 9, |a, j| {
            let (x, z) = (0.1 * (i0 + a) as f64, 0.08 * j as f64);
            DVec3::new(x, 0.3 + 0.15 * (1.3 * x).sin() * (0.9 * z).cos() + 0.05 * x * z, z)
        });
        let u = Grid2::from_fn(i1 - i0, 9, |a, j| DVec2::new(0.1 * (i0 + a) as f64, 0.08 * j as f64));
        (p, u)
    }

    /// Two blocks sharing the i = 5 row: islands 3 and 7.
    fn blocks() -> Vec<(Mesh, i32)> {
        let (pa, ua) = pts(0, 6);
        let (pb, ub) = pts(5, 10);
        vec![
            (grid(&pa, Some(&ua), Mat::Paint, false, false, -1), 3),
            (grid(&pb, Some(&ub), Mat::Paint, false, false, -1), 7),
        ]
    }

    /// Python `summ`: counts, sum(v), sum((k+1) * (x + 2y + 3z)), island
    /// sum, uv sum, material sum, index sum.
    struct Summ {
        nv: usize,
        nq: usize,
        nt: usize,
        vsum: f64,
        wsum: f64,
        isl: i64,
        uvsum: f64,
        mats: i64,
        idx: i64,
    }

    fn summ(m: &Mesh) -> Summ {
        let mut vsum = 0.0;
        let mut wsum = 0.0;
        for (k, p) in m.v.iter().enumerate() {
            vsum += p.x + p.y + p.z;
            wsum += (k + 1) as f64 * (p.x + 2.0 * p.y + 3.0 * p.z);
        }
        Summ {
            nv: m.v.len(),
            nq: m.q.len(),
            nt: m.t.len(),
            vsum,
            wsum,
            isl: m.qi.iter().chain(&m.ti).map(|&x| x as i64).sum(),
            uvsum: m.quv.iter().flatten().chain(m.tuv.iter().flatten()).map(|u| u.x + u.y).sum(),
            mats: m.qm.iter().chain(&m.tm).map(|&x| x as i64).sum(),
            idx: m.q.iter().flatten().chain(m.t.iter().flatten()).map(|&x| x as i64).sum(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check(m: &Mesh, counts: [usize; 3], vsum: f64, wsum: f64, isl: i64, uvsum: f64, mats: i64, idx: i64, probes: &[(usize, [f64; 3])]) {
        let s = summ(m);
        assert_eq!([s.nv, s.nq, s.nt], counts);
        assert_eq!((s.isl, s.mats, s.idx), (isl, mats, idx));
        assert!(m.v.iter().all(|p| p.is_finite()), "NaN in vertices");
        let tol = 1e-9;
        assert!((s.vsum - vsum).abs() < tol * 100.0, "vsum {} vs {}", s.vsum, vsum);
        assert!((s.wsum - wsum).abs() < tol * 1e4, "wsum {} vs {}", s.wsum, wsum);
        assert!((s.uvsum - uvsum).abs() < tol * 100.0, "uvsum {} vs {}", s.uvsum, uvsum);
        for &(i, p) in probes {
            let d = (m.v[i] - DVec3::from(p)).abs().max_element();
            assert!(d < tol, "v[{i}] = {:?} vs {:?} (|d| {d:e})", m.v[i], p);
        }
    }

    fn shut_keys() -> HashSet<Key> {
        let (pa, _) = pts(0, 6);
        let mut s = HashSet::new();
        for a in 0..pa.nu {
            s.insert(key3(pa.at(a, 0), 1e-6));
        }
        for j in 0..pa.nv {
            s.insert(key3(pa.at(0, j), 1e-6));
        }
        s
    }

    #[test]
    fn boundary_loops_3x3_grid() {
        let g = Grid2::from_fn(4, 4, |i, j| DVec3::new(i as f64, j as f64, 0.0));
        let m = grid(&g, None, Mat::Paint, false, false, -1);
        let loops = boundary_loops(&m);
        // Python: [[0, 4, 8, 12, 13, 14, 15, 11, 7, 3, 2, 1]]
        assert_eq!(loops, vec![vec![0, 4, 8, 12, 13, 14, 15, 11, 7, 3, 2, 1]]);
        // consistent orientation: every step is a directed face edge
        let mut fe = HashSet::new();
        for f in m.faces() {
            for i in 0..f.len() {
                fe.insert((f[i], f[(i + 1) % f.len()]));
            }
        }
        let lp = &loops[0];
        for i in 0..lp.len() {
            assert!(fe.contains(&(lp[i], lp[(i + 1) % lp.len()])));
        }
    }

    #[test]
    fn python_floor_division() {
        assert_eq!(py_floordiv(0.0, 1.2), 0.0);
        assert_eq!(py_floordiv(1.2, 1.2), 1.0);
        assert_eq!(py_floordiv(2.4, 1.2), 2.0); // Python: 2.4 // 1.2 == 2.0
        assert_eq!(py_floordiv(3.5999999, 1.2), 2.0);
        assert_eq!(percentile_linear(&[3.0, 1.0, 2.0, 4.0], 1.0), 1.03);
    }

    // Reference numbers below come from vehgen/panels.py (numpy 1.24.3,
    // Blender 4.3's python3.11, run from engine/tools/vehicle-gen) on the
    // same `blocks()` patch, printing len(v), len(q), len(t), v.sum(),
    // (arange(1, n+1) * (v @ [1, 2, 3])).sum(), qi.sum() + ti.sum(),
    // quv.sum(), qm.sum() + tm.sum(), q.sum() + t.sum() and three vertices.

    #[test]
    fn panel_hem_inner_matches_python() {
        // panel(blocks(), inner=0.035)
        let m = panel(&blocks(), 0.0046, true, Some(0.035), 0, None);
        // hem: 2 rows of 34 boundary vertices, 2 quads per boundary edge
        check(
            &m,
            [160, 140, 0],
            184.16570575530397,
            29114.757235124605,
            7196,
            385.5304727653711,
            0,
            39885,
            &[
                (0, [0.023186633728267892, 0.30466883374258574, 0.023662305156774824]),
                (53, [0.4998412296420717, 0.3950130689804659, 0.3199980800852239]),
                (159, [0.4942456800516807, 0.4295628086957299, 0.3199252076694279]),
            ],
        );
    }

    #[test]
    fn panel_nohem_matches_python() {
        // panel(blocks(), hem=False, island0=50)
        let m = panel(&blocks(), 0.0046, false, None, 50, None);
        check(
            &m,
            [90, 72, 0],
            103.49328926846327,
            9817.055258503051,
            3944,
            221.76,
            0,
            12927,
            &[
                (0, [0.021220017987550216, 0.3042692884009936, 0.02164994067972658]),
                (30, [0.29991245630119917, 0.3583343171164432, 0.5588544236676214]),
                (89, [0.8977114044155298, 0.447030361024022, 0.4000002608089192]),
            ],
        );
    }

    #[test]
    fn panel_shut_matches_python() {
        // shut = keys of block A's j=0 column and i=0 row
        let shut = shut_keys();
        let m = panel(&blocks(), 0.0046, true, None, 0, Some(&shut));
        check(
            &m,
            [158, 140, 0],
            181.95819460180275,
            28426.861979516354,
            7198,
            377.88996766621653,
            0,
            39885,
            &[
                (0, [0.023186633728267892, 0.30466883374258574, 0.023662305156774824]),
                (52, [0.49984472678903774, 0.3949331992131356, 0.40000342701822433]),
                (157, [0.009732653109913253, 0.31061380319728465, 0.08245812882417869]),
            ],
        );
    }

    #[test]
    fn glass_matches_python() {
        let m = glass_from(&blocks(), 0.012, 0.006, None);
        check(
            &m,
            [90, 72, 0],
            104.37620646010721,
            9921.236316605971,
            -72,
            221.76,
            288,
            12927,
            &[
                (0, [-0.00819123685626623, 0.31058063653449336, -0.006026656317647825]),
                (30, [0.29781957092206457, 0.3701020867497295, 0.5631167254216788]),
                (89, [0.9047622101465783, 0.45979486962722377, 0.39997634373463015]),
            ],
        );
        let shut = shut_keys();
        let m = glass_from(&blocks(), 0.012, 0.006, Some(&shut));
        check(
            &m,
            [90, 72, 0],
            104.58709772125738,
            9928.896958149046,
            -72,
            221.76,
            288,
            12927,
            &[
                (0, [0.0006301390501462826, 0.3123728910637994, 0.002974774634714333]),
                (30, [0.29781978163153194, 0.3701021237239526, 0.5631169342917841]),
                (89, [0.9047622101465783, 0.45979486962722377, 0.39997634373463015]),
            ],
        );
    }

    #[test]
    fn offset_copy_matches_python() {
        // o = blocks()[0][0]; o.weld(); offset_copy(o, 0.01, mat='chrome', flip=True)
        let mut o = blocks().swap_remove(0).0;
        o.weld(1e-7);
        let m = offset_copy(&o, 0.01, Some(Mat::Chrome), true);
        check(
            &m,
            [54, 40, 0],
            50.00470271108862,
            3078.546216445889,
            -40,
            91.19999999999999,
            40,
            4303,
            &[
                (0, [-0.001915057830387333, 0.3098148315893141, -4.04274618719267e-05]),
                (18, [0.19815494940488587, 0.3483900411078364, -8.380622250333676e-05]),
                (53, [0.4982685017541732, 0.4048881106209967, 0.3199848341936591]),
            ],
        );
    }

    #[test]
    fn inset_and_inner_panel_match_python() {
        // m = A.copy().add(B); m.weld(); loops, isb = inset_boundary(m, full(0.004));
        // inner_panel(m, isb)  (depth 0.035, island 2000, rings 2)
        let bl = blocks();
        let mut m = bl[0].0.clone();
        m.add(&bl[1].0);
        m.weld(1e-7);
        let d = vec![0.004; m.v.len()];
        let (_, isb) = inset_boundary(&mut m, &d, None, None);
        check(
            &m,
            [90, 72, 0],
            103.49904226259954,
            9819.5659307859,
            -72,
            221.76,
            0,
            12927,
            &[
                (0, [0.003919976415656974, 0.30079643132573375, 0.003999899526713212]),
                (30, [0.29999598103788316, 0.35835971613326545, 0.5580001276961346]),
                (89, [0.8960198285026894, 0.44686132062636025, 0.4000004534938803]),
            ],
        );
        let ip = inner_panel(&m, &isb, 0.035, 2000, 2, Mat::Paint);
        check(
            &ip,
            [12, 6, 0],
            14.212034957975238,
            182.08345280748188,
            -6,
            18.480000000000004,
            0,
            138,
            &[
                (0, [0.29364489254184084, 0.39372099441990704, 0.23986150535875697]),
                (4, [0.39395696488739135, 0.41222846452113915, 0.40011147057072566]),
                (11, [0.5949224258028302, 0.44537663510982367, 0.3198811387154919]),
            ],
        );
    }
}
