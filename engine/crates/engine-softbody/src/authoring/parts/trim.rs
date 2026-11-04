//! Cutting the body surface into panels.
//!
//! A region is a signed distance function over the surface's vertices
//! (negative inside), built from smooth cut lines and outlines in whichever
//! view suits them (side view for doors, plan for the bonnet, front view or arc
//! round the outline for lamps). [`clip`] runs marching triangles over the
//! surface mesh: kept triangles are copied, crossed ones are cut exactly where
//! the (linearly interpolated) distance is the cut level, so a panel's edge is
//! the smooth curve itself, never the grid's staircase. The cut's boundary
//! comes back as ordered loops.
//!
//! [`roll`] turns a loop into a real panel edge: the skin rolls over (radius
//! `roll`) into a wall `depth` deep and a return flange under the panel, set
//! back `gap` from the cut line, so two neighbours leave a 2 × `gap` shut line
//! with rounded, catching edges and a dark, occluded slot between them.

use super::Mesh;
use glam::Vec3;
use std::collections::HashMap;

/// A panel edge's construction.
#[derive(Clone, Copy, Debug)]
pub struct EdgeStyle {
    /// Distance from the cut line to the edge's wall (half the shut line), m.
    pub gap: f32,
    /// Roll radius, m.
    pub roll: f32,
    /// Wall depth below the skin, m.
    pub depth: f32,
    /// Return flange width (under the panel), m.
    pub flange: f32,
    /// AO at the bottom of the wall.
    pub ao_floor: f32,
}

impl EdgeStyle {
    /// A door, bonnet or boot shut line (4.5 mm visible gap).
    pub const SHUT: Self = Self {
        gap: 0.0022,
        roll: 0.0028,
        depth: 0.012,
        flange: 0.006,
        ao_floor: 0.12,
    };
    /// An opening the skin turns into (window, lamp): tight, deep.
    pub const OPENING: Self = Self {
        gap: 0.0015,
        roll: 0.003,
        depth: 0.02,
        flange: 0.006,
        ao_floor: 0.2,
    };
    /// A wheel arch lip: a big roll.
    pub const ARCH: Self = Self {
        gap: 0.0,
        roll: 0.008,
        depth: 0.035,
        flange: 0.02,
        ao_floor: 0.15,
    };
    /// A trim strip's edge (a moulding).
    pub const TRIM: Self = Self {
        gap: 0.0,
        roll: 0.002,
        depth: 0.004,
        flange: 0.0,
        ao_floor: 0.5,
    };
}

/// A clipped piece: the mesh, its boundary loops (vertex ids, interior on the
/// left seen from outside) and the cut field interpolated onto its vertices.
pub struct Clipped {
    pub mesh: Mesh,
    pub loops: Vec<(Vec<u32>, bool)>,
    pub field: Vec<f32>,
}

/// Keep the part of `src` where `sdf < level`.
pub fn clip(src: &Mesh, sdf: &[f32], level: f32) -> Clipped {
    let mut out = Mesh::default();
    let mut field = Vec::new();
    let mut vmap: HashMap<u32, u32> = HashMap::new();
    let mut emap: HashMap<(u32, u32), u32> = HashMap::new();
    let mut segs: Vec<(u32, u32)> = Vec::new();
    let val = |i: u32| sdf[i as usize] - level;
    let mut keep = |out: &mut Mesh, field: &mut Vec<f32>, i: u32| -> u32 {
        *vmap.entry(i).or_insert_with(|| {
            let k = i as usize;
            let id = out.vert(src.pos[k], src.nrm[k], src.mat[k], src.ao[k]);
            out.smooth[id as usize] = src.smooth[k];
            field.push(sdf[k]);
            id
        })
    };
    let mut cross = |out: &mut Mesh, field: &mut Vec<f32>, a: u32, b: u32| -> u32 {
        let key = (a.min(b), a.max(b));
        *emap.entry(key).or_insert_with(|| {
            let (a, b) = key;
            let (va, vb) = (val(a), val(b));
            let t = (va / (va - vb)).clamp(0.0, 1.0);
            let (ka, kb) = (a as usize, b as usize);
            let p = src.pos[ka].lerp(src.pos[kb], t);
            let n = src.nrm[ka].lerp(src.nrm[kb], t);
            let id = out.vert(p, n, src.mat[ka], src.ao[ka] + (src.ao[kb] - src.ao[ka]) * t);
            out.smooth[id as usize] = src.smooth[ka];
            field.push(level);
            id
        })
    };
    for t in src.idx.chunks(3) {
        let v = [t[0], t[1], t[2]];
        let ins = v.map(|i| val(i) < 0.0);
        let n_in = ins.iter().filter(|&&b| b).count();
        if n_in == 0 {
            continue;
        }
        if n_in == 3 {
            let a = keep(&mut out, &mut field, v[0]);
            let b = keep(&mut out, &mut field, v[1]);
            let c = keep(&mut out, &mut field, v[2]);
            out.tri(a, b, c);
            continue;
        }
        // Polygon in winding order: kept corners and edge crossings.
        let mut poly: Vec<(u32, bool)> = Vec::with_capacity(4);
        for k in 0..3 {
            let (a, b) = (v[k], v[(k + 1) % 3]);
            if ins[k] {
                poly.push((keep(&mut out, &mut field, a), false));
            }
            if ins[k] != ins[(k + 1) % 3] {
                poly.push((cross(&mut out, &mut field, a, b), true));
            }
        }
        for k in 1..poly.len() - 1 {
            out.tri(poly[0].0, poly[k].0, poly[k + 1].0);
        }
        // The contour edge: between the two crossings, in polygon order.
        for k in 0..poly.len() {
            let (p, q) = (poly[k], poly[(k + 1) % poly.len()]);
            if p.1 && q.1 {
                segs.push((p.0, q.0));
            }
        }
    }
    let loops = chain(&segs);
    Clipped {
        mesh: out,
        loops,
        field,
    }
}

/// Chain directed segments into loops (closed) and chains (open).
fn chain(segs: &[(u32, u32)]) -> Vec<(Vec<u32>, bool)> {
    let mut next: HashMap<u32, u32> = HashMap::new();
    let mut has_prev: HashMap<u32, bool> = HashMap::new();
    for &(a, b) in segs {
        next.insert(a, b);
        has_prev.insert(b, true);
    }
    let mut used: HashMap<u32, bool> = HashMap::new();
    let mut out = Vec::new();
    // Open chains first (start where nothing leads in).
    let mut starts: Vec<u32> = segs
        .iter()
        .map(|s| s.0)
        .filter(|a| !has_prev.contains_key(a))
        .collect();
    starts.sort_unstable();
    for s in starts {
        let mut l = vec![s];
        used.insert(s, true);
        let mut c = s;
        while let Some(&n) = next.get(&c) {
            if used.contains_key(&n) {
                break;
            }
            used.insert(n, true);
            l.push(n);
            c = n;
        }
        if l.len() > 1 {
            out.push((l, false));
        }
    }
    let mut keys: Vec<u32> = segs.iter().map(|s| s.0).collect();
    keys.sort_unstable();
    for s in keys {
        if used.contains_key(&s) {
            continue;
        }
        let mut l = vec![s];
        used.insert(s, true);
        let mut c = s;
        while let Some(&n) = next.get(&c) {
            if used.contains_key(&n) {
                break;
            }
            used.insert(n, true);
            l.push(n);
            c = n;
        }
        if l.len() > 2 {
            out.push((l, true));
        }
    }
    out
}

/// Keep where `lo < field < hi` (a ring or strip between two levels).
pub fn clip_between(src: &Mesh, sdf: &[f32], lo: f32, hi: f32) -> Clipped {
    let a = clip(src, sdf, hi);
    let neg: Vec<f32> = a.field.iter().map(|v| -v).collect();
    let b = clip(&a.mesh, &neg, -lo);
    let field = b.field.iter().map(|v| -v).collect();
    // Loops of the second clip only cover the low edge; recompute both by
    // chaining the open boundary of the result.
    let loops = boundary_loops(&b.mesh);
    Clipped {
        mesh: b.mesh,
        loops,
        field,
    }
}

/// The boundary loops of a mesh (edges used by one triangle), oriented with
/// the interior on the left.
pub fn boundary_loops(m: &Mesh) -> Vec<(Vec<u32>, bool)> {
    let mut count: HashMap<(u32, u32), i32> = HashMap::new();
    for t in m.idx.chunks(3) {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    // Weld coincident vertices for the count (clipping duplicates none, so ids
    // suffice).
    let mut segs = Vec::new();
    for t in m.idx.chunks(3) {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if count[&(a.min(b), a.max(b))] == 1 {
                segs.push((a, b));
            }
        }
    }
    chain(&segs)
}

/// Darken a clipped piece toward its cut (the panel's edge AO): `level` the
/// cut level, `width` the falloff (m).
pub fn edge_ao(c: &mut Clipped, level: f32, strength: f32, width: f32) {
    for (a, f) in c.mesh.ao.iter_mut().zip(&c.field) {
        let d = (level - f).max(0.0);
        *a *= 1.0 - strength * (-d / width).exp();
    }
}

/// Roll every loop of `c` into a panel edge.
pub fn roll(c: &mut Clipped, st: EdgeStyle) {
    let loops = std::mem::take(&mut c.loops);
    for (l, closed) in &loops {
        roll_loop(&mut c.mesh, l, *closed, st);
    }
    c.loops = loops;
}

pub fn roll_loop(m: &mut Mesh, l: &[u32], closed: bool, st: EdgeStyle) {
    let n = l.len();
    if n < 2 {
        return;
    }
    let steps = 3usize;
    let mut rings: Vec<Vec<u32>> = vec![l.to_vec()];
    let mut frames = Vec::with_capacity(n);
    for k in 0..n {
        let (a, b) = if closed {
            (l[(k + n - 1) % n], l[(k + 1) % n])
        } else {
            (l[k.saturating_sub(1)], l[(k + 1).min(n - 1)])
        };
        let t = (m.pos[b as usize] - m.pos[a as usize]).normalize_or_zero();
        let nn = m.nrm[l[k] as usize];
        // Interior on the left (seen from outside): outward = t × n.
        let o = t.cross(nn).normalize_or_zero();
        let nn = o.cross(t).normalize_or_zero();
        frames.push((m.pos[l[k] as usize], nn, o, m.mat[l[k] as usize], m.ao[l[k] as usize], m.smooth[l[k] as usize]));
    }
    let r = st.roll;
    let g = st.gap;
    let _ = g;
    for s in 1..=steps {
        let phi = std::f32::consts::FRAC_PI_2 * s as f32 / steps as f32;
        let ring = frames
            .iter()
            .map(|&(p, nn, o, mat, ao, sm)| {
                let q = p + o * (r * phi.sin()) - nn * (r * (1.0 - phi.cos()));
                let id = m.vert(q, nn * phi.cos() + o * phi.sin(), mat, ao * (1.0 - 0.35 * s as f32 / steps as f32));
                m.smooth[id as usize] = sm;
                id
            })
            .collect();
        rings.push(ring);
    }
    // Wall and flange.
    let wall = frames
        .iter()
        .map(|&(p, nn, o, mat, _, sm)| {
            let id = m.vert(p + o * r - nn * (r + st.depth), o, mat, st.ao_floor);
            m.smooth[id as usize] = sm;
            id
        })
        .collect();
    rings.push(wall);
    if st.flange > 0.0 {
        let fl = frames
            .iter()
            .map(|&(p, nn, o, mat, _, sm)| {
                let id = m.vert(p + o * (r - st.flange) - nn * (r + st.depth), -nn, mat, st.ao_floor * 0.6);
                m.smooth[id as usize] = sm;
                id
            })
            .collect();
        rings.push(fl);
    }
    // Quads: ring k → k + 1 along the loop, wound to face outward (o side).
    for w in rings.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let segs = if closed { n } else { n - 1 };
        for i in 0..segs {
            let j = (i + 1) % n;
            // Loop direction t, ring direction toward −n: face normal ≈ o.
            m.quad(a[i], b[i], b[j], a[j]);
        }
    }
}

/// A copy of `m` moved `d` along its normals (flipped winding and normals when
/// `flip`), with material `mat` (or the source's) and AO scaled.
pub fn offset(m: &Mesh, d: f32, flip: bool, mat: Option<u8>, ao: f32) -> Mesh {
    let mut o = m.clone();
    for (p, n) in o.pos.iter_mut().zip(o.nrm.iter_mut()) {
        *p += *n * d;
        if flip {
            *n = -*n;
        }
    }
    if let Some(mt) = mat {
        o.mat.iter_mut().for_each(|x| *x = mt);
    }
    o.ao.iter_mut().for_each(|a| *a *= ao);
    if flip {
        for t in o.idx.chunks_mut(3) {
            t.swap(1, 2);
        }
    }
    o
}

/// Evaluate a region on every vertex of `m`.
pub fn field(m: &Mesh, f: impl Fn(usize, Vec3) -> f32) -> Vec<f32> {
    m.pos.iter().enumerate().map(|(i, &p)| f(i, p)).collect()
}
