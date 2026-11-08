//! A small mesh container (quads + triangles, per-face material and
//! per-corner UVs) and the primitive generators the parts are built from.
//!
//! Coordinates are glTF's: metres, +Y up, the car's front +Z, +X the car's
//! left. A line-for-line port of `vehgen/mesh.py`, including numpy's
//! conventions where they decide vertex order (`np.unique` sorts keys,
//! `np.round` rounds half to even), so the two generators can be compared.

use std::cell::Cell;

use glam::{DMat3, DVec2, DVec3};

/// The art spec's ten materials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Mat {
    Paint = 0,
    Chrome,
    TrimBlack,
    Rubber,
    Glass,
    LampLens,
    LampEmissive,
    InteriorFabric,
    InteriorPlastic,
    MetalBare,
}

pub const MATERIALS: [&str; 10] = [
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

impl Mat {
    pub fn name(self) -> &'static str {
        MATERIALS[self as usize]
    }
    pub fn all() -> [Mat; 10] {
        use Mat::*;
        [Paint, Chrome, TrimBlack, Rubber, Glass, LampLens, LampEmissive, InteriorFabric, InteriorPlastic, MetalBare]
    }
}

thread_local! {
    static ISLAND: Cell<i32> = const { Cell::new(100000) };
}

/// A fresh UV island id (swept and lofted parts each get their own).
pub fn new_island() -> i32 {
    ISLAND.with(|c| {
        c.set(c.get() + 1);
        c.get()
    })
}

/// Start island numbering over (each build starts from the same state).
pub fn reset_islands() {
    ISLAND.with(|c| c.set(100000));
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub v: Vec<DVec3>,
    pub q: Vec<[u32; 4]>,
    pub qm: Vec<Mat>,
    pub quv: Vec<[DVec2; 4]>,
    /// uv island id per quad (-1: none)
    pub qi: Vec<i32>,
    pub t: Vec<[u32; 3]>,
    pub tm: Vec<Mat>,
    pub tuv: Vec<[DVec2; 3]>,
    pub ti: Vec<i32>,
    /// per-vertex colour (lamp lens tints) or None
    pub col: Option<Vec<[f64; 4]>>,
}

/// Round half to even, as `np.round` does.
#[inline]
pub fn round_key(x: f64, tol: f64) -> i64 {
    (x / tol).round_ties_even() as i64
}

#[inline]
pub fn key3(p: DVec3, tol: f64) -> (i64, i64, i64) {
    (round_key(p.x, tol), round_key(p.y, tol), round_key(p.z, tol))
}

impl Mesh {
    pub fn new() -> Mesh {
        Mesh::default()
    }

    pub fn from_faces(
        v: Vec<DVec3>,
        quads: Vec<[u32; 4]>,
        tris: Vec<[u32; 3]>,
        mat: Mat,
        quv: Option<Vec<[DVec2; 4]>>,
        island: i32,
    ) -> Mesh {
        let nq = quads.len();
        let nt = tris.len();
        Mesh {
            v,
            qm: vec![mat; nq],
            quv: quv.unwrap_or_else(|| vec![[DVec2::ZERO; 4]; nq]),
            qi: vec![island; nq],
            q: quads,
            tm: vec![mat; nt],
            tuv: vec![[DVec2::ZERO; 3]; nt],
            ti: vec![island; nt],
            t: tris,
            col: None,
        }
    }

    pub fn add(&mut self, o: &Mesh) -> &mut Mesh {
        let n = self.v.len() as u32;
        if self.col.is_some() || o.col.is_some() {
            let mut a = self.col.take().unwrap_or_else(|| vec![[1.0; 4]; self.v.len()]);
            match &o.col {
                Some(b) => a.extend_from_slice(b),
                None => a.extend(std::iter::repeat_n([1.0; 4], o.v.len())),
            }
            self.col = Some(a);
        }
        self.v.extend_from_slice(&o.v);
        self.q.extend(o.q.iter().map(|f| f.map(|i| i + n)));
        self.qm.extend_from_slice(&o.qm);
        self.quv.extend_from_slice(&o.quv);
        self.qi.extend_from_slice(&o.qi);
        self.t.extend(o.t.iter().map(|f| f.map(|i| i + n)));
        self.tm.extend_from_slice(&o.tm);
        self.tuv.extend_from_slice(&o.tuv);
        self.ti.extend_from_slice(&o.ti);
        self
    }

    /// `add` taking ownership (chains like the Python `m.add(a).add(b)`).
    pub fn with(mut self, o: Mesh) -> Mesh {
        self.add(&o);
        self
    }

    pub fn set_mat(&mut self, mat: Mat) -> &mut Mesh {
        self.qm.iter_mut().for_each(|m| *m = mat);
        self.tm.iter_mut().for_each(|m| *m = mat);
        self
    }

    pub fn tint(&mut self, rgba: [f64; 4]) -> &mut Mesh {
        self.col = Some(vec![rgba; self.v.len()]);
        self
    }

    pub fn transform(&mut self, r: Option<DMat3>, t: DVec3) -> &mut Mesh {
        if let Some(r) = r {
            for p in &mut self.v {
                *p = r * *p;
            }
            if r.determinant() < 0.0 {
                self.flip();
            }
        }
        for p in &mut self.v {
            *p += t;
        }
        self
    }

    pub fn translate(&mut self, t: DVec3) -> &mut Mesh {
        self.transform(None, t)
    }

    pub fn mirror_x(&mut self) -> &mut Mesh {
        for p in &mut self.v {
            p.x = -p.x;
        }
        self.flip()
    }

    pub fn flip(&mut self) -> &mut Mesh {
        for f in &mut self.q {
            f.reverse();
        }
        for u in &mut self.quv {
            u.reverse();
        }
        for f in &mut self.t {
            f.reverse();
        }
        for u in &mut self.tuv {
            u.reverse();
        }
        self
    }

    pub fn ntris(&self) -> usize {
        2 * self.q.len() + self.t.len()
    }

    /// All faces as triangles: the triangles, then every quad's (0,1,2),
    /// then every quad's (0,2,3) (the Python order).
    pub fn triangles(&self) -> Vec<[u32; 3]> {
        let mut out = self.t.clone();
        out.extend(self.q.iter().map(|f| [f[0], f[1], f[2]]));
        out.extend(self.q.iter().map(|f| [f[0], f[2], f[3]]));
        out
    }

    /// Merge coincident vertices (exactly shared edge samples). Vertex order
    /// follows `np.unique`: sorted by the rounded key.
    pub fn weld(&mut self, tol: f64) -> &mut Mesh {
        let keys: Vec<(i64, i64, i64)> = self.v.iter().map(|p| key3(*p, tol)).collect();
        let mut order: Vec<usize> = (0..self.v.len()).collect();
        // stable sort: equal keys keep their original order, so the first
        // index of each run is the first occurrence (np.unique return_index)
        order.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
        let mut inv = vec![0u32; self.v.len()];
        let mut first: Vec<usize> = Vec::new();
        for (k, &i) in order.iter().enumerate() {
            if k == 0 || keys[i] != keys[order[k - 1]] {
                first.push(i);
            }
            inv[i] = (first.len() - 1) as u32;
        }
        self.v = first.iter().map(|&i| self.v[i]).collect();
        if let Some(c) = &self.col {
            self.col = Some(first.iter().map(|&i| c[i]).collect());
        }
        for f in &mut self.q {
            *f = f.map(|i| inv[i as usize]);
        }
        for f in &mut self.t {
            *f = f.map(|i| inv[i as usize]);
        }
        // drop collapsed faces; quads with one collapsed edge become triangles
        let mut keepq = vec![true; self.q.len()];
        for i in 0..self.q.len() {
            let f = self.q[i];
            let mut u = f.to_vec();
            u.sort_unstable();
            u.dedup();
            if u.len() == 4 {
                continue;
            }
            keepq[i] = false;
            if u.len() == 3 {
                for k in 0..4 {
                    if f[k] == f[(k + 1) % 4] {
                        let idx: Vec<usize> = (0..4).filter(|&j| j != (k + 1) % 4).collect();
                        self.t.push([f[idx[0]], f[idx[1]], f[idx[2]]]);
                        self.tm.push(self.qm[i]);
                        let uv = self.quv[i];
                        self.tuv.push([uv[idx[0]], uv[idx[1]], uv[idx[2]]]);
                        self.ti.push(self.qi[i]);
                        break;
                    }
                }
            }
        }
        let mut k = 0;
        self.q.retain(|_| {
            k += 1;
            keepq[k - 1]
        });
        let mut k = 0;
        self.qm.retain(|_| {
            k += 1;
            keepq[k - 1]
        });
        let mut k = 0;
        self.quv.retain(|_| {
            k += 1;
            keepq[k - 1]
        });
        let mut k = 0;
        self.qi.retain(|_| {
            k += 1;
            keepq[k - 1]
        });
        let kt: Vec<bool> = self.t.iter().map(|f| f[0] != f[1] && f[1] != f[2] && f[0] != f[2]).collect();
        retain_by(&mut self.t, &kt);
        retain_by(&mut self.tm, &kt);
        retain_by(&mut self.tuv, &kt);
        retain_by(&mut self.ti, &kt);
        self
    }

    pub fn vertex_normals(&self) -> Vec<DVec3> {
        let mut n = vec![DVec3::ZERO; self.v.len()];
        for f in self.triangles() {
            let (a, b, c) = (self.v[f[0] as usize], self.v[f[1] as usize], self.v[f[2] as usize]);
            let fnrm = (b - a).cross(c - a);
            for &i in &f {
                n[i as usize] += fnrm;
            }
        }
        n.iter().map(|x| *x / x.length().max(1e-12)).collect()
    }

    pub fn bbox(&self) -> (DVec3, DVec3) {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in &self.v {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (lo, hi)
    }

    /// Faces as index lists (quads then triangles), the Python `faces_list`.
    pub fn faces(&self) -> Vec<Vec<u32>> {
        self.q.iter().map(|f| f.to_vec()).chain(self.t.iter().map(|f| f.to_vec())).collect()
    }

    pub fn set_islands(&mut self, id: i32) {
        self.qi.iter_mut().for_each(|x| *x = id);
        self.ti.iter_mut().for_each(|x| *x = id);
    }

    /// Append triangles of one material with no UVs (caps).
    pub fn push_tris(&mut self, tris: &[[u32; 3]], mat: Mat) {
        for t in tris {
            self.t.push(*t);
            self.tm.push(mat);
            self.tuv.push([DVec2::ZERO; 3]);
            self.ti.push(-1);
        }
    }
}

pub fn retain_by<T>(v: &mut Vec<T>, keep: &[bool]) {
    let mut k = 0;
    v.retain(|_| {
        k += 1;
        keep[k - 1]
    });
}

// -- primitives -----------------------------------------------------------------

/// A (nu x nv) array of points, row-major (u outer).
#[derive(Clone, Debug)]
pub struct Grid2<T> {
    pub nu: usize,
    pub nv: usize,
    pub d: Vec<T>,
}

impl<T: Copy> Grid2<T> {
    pub fn new(nu: usize, nv: usize, d: Vec<T>) -> Self {
        assert_eq!(d.len(), nu * nv);
        Grid2 { nu, nv, d }
    }
    pub fn from_fn(nu: usize, nv: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut d = Vec::with_capacity(nu * nv);
        for i in 0..nu {
            for j in 0..nv {
                d.push(f(i, j));
            }
        }
        Grid2 { nu, nv, d }
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> T {
        self.d[i * self.nv + j]
    }
}

/// Quad grid from points P (nu, nv); faces wound so (dP/du x dP/dv) is the
/// front. `uv` may carry the seam column/row explicitly.
pub fn grid(p: &Grid2<DVec3>, uv: Option<&Grid2<DVec2>>, mat: Mat, closed_u: bool, closed_v: bool, island: i32) -> Mesh {
    let (nu, nv) = (p.nu, p.nv);
    let ru = if closed_u { nu + 1 } else { nu };
    let rv = if closed_v { nv + 1 } else { nv };
    let idx = |i: usize, j: usize| -> u32 { ((i % nu) * nv + (j % nv)) as u32 };
    let mut quads = Vec::with_capacity((ru - 1) * (rv - 1));
    let mut quv = Vec::with_capacity((ru - 1) * (rv - 1));
    // uv extended with the seam if it was given without one
    let uvx = uv.map(|u| {
        let (unu, unv) = (u.nu, u.nv);
        Grid2::from_fn(ru, rv, |i, j| {
            let ii = if i >= unu { i - unu } else { i };
            let jj = if j >= unv { j - unv } else { j };
            u.at(ii, jj)
        })
    });
    for i in 0..ru - 1 {
        for j in 0..rv - 1 {
            quads.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            if let Some(u) = &uvx {
                quv.push([u.at(i, j), u.at(i + 1, j), u.at(i + 1, j + 1), u.at(i, j + 1)]);
            }
        }
    }
    Mesh::from_faces(p.d.clone(), quads, vec![], mat, if uv.is_some() { Some(quv) } else { None }, island)
}

/// Revolve profile [(r, a), ...] about the +X axis (a = axial position).
/// Points with r == 0 become poles.
pub fn lathe(profile: &[(f64, f64)], segs: usize, mat: Mat, a0: f64, a1: f64) -> Mesh {
    let full = ((a1 - a0) - 2.0 * std::f64::consts::PI).abs() < 1e-9;
    let n = if full { segs } else { segs + 1 };
    let th: Vec<f64> = (0..n).map(|k| a0 + (a1 - a0) * k as f64 / segs as f64).collect();
    let mut verts: Vec<DVec3> = Vec::new();
    let mut ids = vec![vec![0u32; n]; profile.len()];
    for (i, &(r, a)) in profile.iter().enumerate() {
        if r < 1e-9 {
            let id = verts.len() as u32;
            ids[i].iter_mut().for_each(|x| *x = id);
            verts.push(DVec3::new(a, 0.0, 0.0));
        } else {
            for (j, &t) in th.iter().enumerate() {
                ids[i][j] = verts.len() as u32;
                verts.push(DVec3::new(a, r * t.cos(), r * t.sin()));
            }
        }
    }
    let mut quads = Vec::new();
    let mut tris = Vec::new();
    let jn = if full { n } else { n - 1 };
    for i in 0..profile.len() - 1 {
        for j in 0..jn {
            let j2 = (j + 1) % n;
            let f = [ids[i][j], ids[i][j2], ids[i + 1][j2], ids[i + 1][j]];
            let mut u: Vec<u32> = Vec::new();
            for x in f {
                if !u.contains(&x) {
                    u.push(x);
                }
            }
            if u.len() == 4 {
                quads.push(f);
            } else if u.len() == 3 {
                tris.push([u[0], u[1], u[2]]);
            }
        }
    }
    Mesh::from_faces(verts, quads, tris, mat, None, -1)
}

pub fn lathe_full(profile: &[(f64, f64)], segs: usize, mat: Mat) -> Mesh {
    lathe(profile, segs, mat, 0.0, 2.0 * std::f64::consts::PI)
}

/// Revolve a closed profile loop (not repeating the first point) into a
/// closed solid, outward facing.
pub fn closed_lathe(profile: &[(f64, f64)], segs: usize, mat: Mat) -> Mesh {
    closed_lathe_arc(profile, segs, mat, 0.0, 2.0 * std::f64::consts::PI)
}

pub fn closed_lathe_arc(profile: &[(f64, f64)], segs: usize, mat: Mat, a0: f64, a1: f64) -> Mesh {
    let mut pr = profile.to_vec();
    pr.push(profile[0]);
    let mut m = lathe(&pr, segs, mat, a0, a1);
    m.weld(1e-7);
    orient_outward(m)
}

/// Closed rounded box: a cube grid pushed onto the Minkowski sum of a box
/// and a sphere.
pub fn rbox(center: DVec3, half: DVec3, radius: f64, n: usize, mat: Mat) -> Mesh {
    let r = radius.min(half.min_element() * 0.999);
    let inner = half - DVec3::splat(r);
    let mut m = Mesh::new();
    let div = 2 * (n + 1);
    for axis in 0..3 {
        for sgn in [-1.0f64, 1.0] {
            let others: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
            let (mut u_ax, mut v_ax) = (others[0], others[1]);
            // (u, v) must be right-handed about the face's outward normal:
            // e_u x e_v = sgn e_axis (for the y faces the pair is (z, x))
            if axis == 1 {
                std::mem::swap(&mut u_ax, &mut v_ax);
            }
            if sgn < 0.0 {
                std::mem::swap(&mut u_ax, &mut v_ax);
            }
            let (nu, nv) = (div, div);
            let cl = |x: f64| x.signum() * x.abs().powf(0.6) * if x == 0.0 { 0.0 } else { 1.0 };
            let us: Vec<f64> = crate::curves::linspace(-1.0, 1.0, nu + 1).into_iter().map(cl).collect();
            let vs: Vec<f64> = crate::curves::linspace(-1.0, 1.0, nv + 1).into_iter().map(cl).collect();
            let g = Grid2::from_fn(nu + 1, nv + 1, |i, j| {
                let mut p = [0.0f64; 3];
                p[axis] = sgn;
                p[u_ax] = us[i];
                p[v_ax] = vs[j];
                let p = DVec3::from_array(p) * half;
                let c = p.clamp(-inner, inner);
                let d = p - c;
                c + d / d.length().max(1e-12) * r
            });
            m.add(&grid(&g, None, mat, false, false, -1));
        }
    }
    m.weld(1e-6);
    m.translate(center);
    m
}

/// Closed axis-aligned box (optionally rotated by R about its centre).
pub fn box_(center: DVec3, half: DVec3, mat: Mat, r: Option<DMat3>) -> Mesh {
    let s: [[f64; 3]; 8] = [
        [-1., -1., -1.],
        [1., -1., -1.],
        [1., 1., -1.],
        [-1., 1., -1.],
        [-1., -1., 1.],
        [1., -1., 1.],
        [1., 1., 1.],
        [-1., 1., 1.],
    ];
    let v: Vec<DVec3> = s
        .iter()
        .map(|a| {
            let p = DVec3::from_array(*a) * half;
            r.map(|r| r * p).unwrap_or(p) + center
        })
        .collect();
    let q = vec![[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]];
    orient_outward(Mesh::from_faces(v, q, vec![], mat, None, -1))
}

/// Options for `sweep`.
#[derive(Clone)]
pub struct SweepOpts {
    pub closed: bool,
    pub caps: bool,
    pub up: DVec3,
    pub scale: Option<Vec<DVec2>>,
    pub twist_up: Option<Vec<DVec3>>,
}

impl Default for SweepOpts {
    fn default() -> Self {
        SweepOpts { closed: false, caps: true, up: DVec3::Y, scale: None, twist_up: None }
    }
}

/// numpy.gradient along a polyline (central inside, one-sided at the ends).
pub fn gradient3(p: &[DVec3]) -> Vec<DVec3> {
    let n = p.len();
    (0..n)
        .map(|i| {
            if i == 0 {
                p[1] - p[0]
            } else if i == n - 1 {
                p[n - 1] - p[n - 2]
            } else {
                (p[i + 1] - p[i - 1]) * 0.5
            }
        })
        .collect()
}

pub fn gradient2(p: &[DVec2]) -> Vec<DVec2> {
    let n = p.len();
    (0..n)
        .map(|i| {
            if i == 0 {
                p[1] - p[0]
            } else if i == n - 1 {
                p[n - 1] - p[n - 2]
            } else {
                (p[i + 1] - p[i - 1]) * 0.5
            }
        })
        .collect()
}

/// Sweep a closed 2D section (counter-clockwise) along a 3D path. sx follows
/// the frame's side vector, sy its up vector. Returns a closed tube (caps by
/// ear clipping) unless `closed`.
pub fn sweep(path: &[DVec3], section: &[DVec2], mat: Mat, o: &SweepOpts) -> Mesh {
    let n = path.len();
    let ns = section.len();
    let mut tg: Vec<DVec3> = if o.closed {
        (0..n).map(|i| path[(i + 1) % n] - path[(i + n - 1) % n]).collect()
    } else {
        gradient3(path)
    };
    for t in &mut tg {
        *t /= t.length();
    }
    let ups: Vec<DVec3> = o.twist_up.clone().unwrap_or_else(|| vec![o.up; n]);
    let mut side: Vec<DVec3> = (0..n).map(|i| tg[i].cross(ups[i])).collect();
    for i in 0..n {
        if side[i].length() < 1e-6 {
            side[i] = tg[i].cross(DVec3::X);
        }
        let l = side[i].length();
        side[i] /= l;
    }
    let upv: Vec<DVec3> = (0..n).map(|i| side[i].cross(tg[i])).collect();
    let sc = o.scale.clone().unwrap_or_else(|| vec![DVec2::ONE; n]);
    let g = Grid2::from_fn(n, ns, |i, k| path[i] + side[i] * (section[k].x * sc[i].x) + upv[i] * (section[k].y * sc[i].y));
    let uv = tube_uv(&g, o.closed);
    let mut m = grid(&g, Some(&uv), mat, o.closed, true, new_island());
    if o.caps && !o.closed {
        for (end, sgn) in [(0usize, -1.0), (n - 1, 1.0)] {
            let ring: Vec<u32> = (0..ns).map(|k| (k + end * ns) as u32).collect();
            let tris = zigzag_cap(&ring, &m.v, tg[end] * sgn);
            m.push_tris(&tris, mat);
        }
    }
    m
}

/// Triangulate a ring (indices) by ear clipping, always cutting the
/// best-shaped ear.
pub fn zigzag_cap(ring: &[u32], p: &[DVec3], outward: DVec3) -> Vec<[u32; 3]> {
    let pts3: Vec<DVec3> = ring.iter().map(|&i| p[i as usize]).collect();
    let c = pts3.iter().copied().sum::<DVec3>() / pts3.len() as f64;
    let nrm = outward / outward.length().max(1e-12);
    let mut ax = nrm.cross(DVec3::X);
    if ax.length() < 0.1 {
        ax = nrm.cross(DVec3::Y);
    }
    ax /= ax.length();
    let ay = nrm.cross(ax);
    let q: Vec<DVec2> = pts3.iter().map(|x| DVec2::new((*x - c).dot(ax), (*x - c).dot(ay))).collect();
    let k = q.len();
    let mut area = 0.0;
    for i in 0..k {
        let j = (i + 1) % k;
        area += q[i].x * q[j].y - q[j].x * q[i].y;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    if area < 0.0 {
        idx.reverse();
    }
    let mut tris: Vec<[u32; 3]> = Vec::new();
    while idx.len() > 3 {
        let mut best = -1.0;
        let mut bi: Option<usize> = None;
        let kk = idx.len();
        for i in 0..kk {
            let a = idx[(i + kk - 1) % kk];
            let b = idx[i];
            let cc = idx[(i + 1) % kk];
            let (aa, bb, c2) = (q[a], q[b], q[cc]);
            let cr = (bb.x - aa.x) * (c2.y - aa.y) - (bb.y - aa.y) * (c2.x - aa.x);
            if cr <= 1e-10 {
                continue;
            }
            let mut ok = true;
            for &j in &idx {
                if j == a || j == b || j == cc {
                    continue;
                }
                let pq = q[j];
                let d1 = (bb.x - aa.x) * (pq.y - aa.y) - (bb.y - aa.y) * (pq.x - aa.x);
                let d2 = (c2.x - bb.x) * (pq.y - bb.y) - (c2.y - bb.y) * (pq.x - bb.x);
                let d3 = (aa.x - c2.x) * (pq.y - c2.y) - (aa.y - c2.y) * (pq.x - c2.x);
                if d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let l = (bb - aa).length().max((c2 - bb).length()).max((aa - c2).length());
            let qual = cr / (l * l);
            if qual > best {
                best = qual;
                bi = Some(i);
            }
        }
        let bi = bi.unwrap_or(1);
        let kk = idx.len();
        let (a, b, cc) = (idx[(bi + kk - 1) % kk], idx[bi], idx[(bi + 1) % kk]);
        tris.push([ring[a], ring[b], ring[cc]]);
        idx.remove(bi);
    }
    tris.push([ring[idx[0]], ring[idx[1]], ring[idx[2]]]);
    let mut nsum = DVec3::ZERO;
    for t in &tris {
        let (a, b, cc) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
        nsum += (b - a).cross(cc - a);
    }
    if nsum.dot(nrm) < 0.0 {
        for t in &mut tris {
            t.reverse();
        }
    }
    tris
}

/// Metric unwrap of a ring grid G (n, m): u along the path, v round the
/// ring, with the seam column (and row) carried past the end.
pub fn tube_uv(g: &Grid2<DVec3>, closed_u: bool) -> Grid2<DVec2> {
    let (n, m) = (g.nu, g.nv);
    let rn = if closed_u { n + 1 } else { n };
    let gc = |i: usize, j: usize| g.at(i % n, j % m);
    let mut du = vec![0.0; rn - 1];
    for (i, d) in du.iter_mut().enumerate() {
        let mut s = 0.0;
        for j in 0..=m {
            s += (gc(i + 1, j) - gc(i, j)).length();
        }
        *d = s / (m + 1) as f64;
    }
    let mut dv = vec![0.0; m];
    for (j, d) in dv.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..rn {
            s += (gc(i, j + 1) - gc(i, j)).length();
        }
        *d = s / rn as f64;
    }
    let mut u = vec![0.0];
    for d in du {
        u.push(u.last().unwrap() + d);
    }
    let mut v = vec![0.0];
    for d in dv {
        v.push(v.last().unwrap() + d);
    }
    Grid2::from_fn(rn, m + 1, |i, j| DVec2::new(u[i], v[j]))
}

pub fn circle(r: f64, n: usize, cx: f64, cy: f64) -> Vec<DVec2> {
    (0..n)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            DVec2::new(cx + r * a.cos(), cy + r * a.sin())
        })
        .collect()
}

/// Counter-clockwise rounded rectangle (w, h full sizes), corner arcs n.
pub fn rrect(w: f64, h: f64, r: f64, n: usize) -> Vec<DVec2> {
    let r = r.min(w / 2.0 * 0.999).min(h / 2.0 * 0.999);
    let mut pts = Vec::new();
    for (cx, cy, a0) in [(w / 2.0 - r, h / 2.0 - r, 0.0), (-w / 2.0 + r, h / 2.0 - r, 90.0), (-w / 2.0 + r, -h / 2.0 + r, 180.0), (w / 2.0 - r, -h / 2.0 + r, 270.0)] {
        for k in 0..=n {
            let a = ((a0 + 90.0 * k as f64 / n as f64) as f64).to_radians();
            pts.push(DVec2::new(cx + r * a.cos(), cy + r * a.sin()));
        }
    }
    pts
}

/// Loft closed 3D rings (k rings of m points) into a closed solid.
pub fn loft(sections: &Grid2<DVec3>, mat: Mat, caps: bool) -> Mesh {
    let (k, mc) = (sections.nu, sections.nv);
    let uv = tube_uv(sections, false);
    let mut m = grid(sections, Some(&uv), mat, false, true, new_island());
    if caps {
        for (end, _sgn) in [(0usize, 1.0), (k - 1, -1.0)] {
            let ring: Vec<u32> = (0..mc).map(|i| (i + end * mc) as u32).collect();
            let other = if end == 0 { 1 } else { k - 2 };
            let mean = |r: usize| (0..mc).map(|j| sections.at(r, j)).sum::<DVec3>() / mc as f64;
            let outward = mean(end) - mean(other);
            let tris = zigzag_cap(&ring, &m.v, outward);
            m.push_tris(&tris, mat);
        }
    }
    m
}

/// Flip a closed mesh so its signed volume is positive.
pub fn orient_outward(mut m: Mesh) -> Mesh {
    let mut vol = 0.0;
    for f in m.triangles() {
        let (a, b, c) = (m.v[f[0] as usize], m.v[f[1] as usize], m.v[f[2] as usize]);
        vol += a.dot(b.cross(c));
    }
    if vol < 0.0 {
        m.flip();
    }
    m
}

pub fn rot_x(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    DMat3::from_cols(DVec3::new(1., 0., 0.), DVec3::new(0., c, s), DVec3::new(0., -s, c))
}

pub fn rot_y(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    DMat3::from_cols(DVec3::new(c, 0., -s), DVec3::new(0., 1., 0.), DVec3::new(s, 0., c))
}

pub fn rot_z(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    DMat3::from_cols(DVec3::new(c, s, 0.), DVec3::new(-s, c, 0.), DVec3::new(0., 0., 1.))
}

/// Per-face box projection; island per dominant axis and sign.
pub fn cube_uv(m: &mut Mesh, island_base: i32, scale: f64) {
    let proj = |p: DVec3, ax: usize| -> DVec2 {
        let o: Vec<usize> = (0..3).filter(|&k| k != ax).collect();
        let a = p.to_array();
        DVec2::new(a[o[0]] * scale, a[o[1]] * scale)
    };
    let pick = |n: DVec3| -> (usize, i32) {
        let a = n.abs().to_array();
        let mut ax = 0;
        for k in 1..3 {
            if a[k] > a[ax] {
                ax = k;
            }
        }
        (ax, if n.to_array()[ax] > 0.0 { 1 } else { 0 })
    };
    for i in 0..m.q.len() {
        let f = m.q[i];
        let p: Vec<DVec3> = f.iter().map(|&k| m.v[k as usize]).collect();
        let n = (p[1] - p[0]).cross(p[2] - p[0]) + (p[2] - p[0]).cross(p[3] - p[0]);
        let (ax, sg) = pick(n);
        m.quv[i] = [proj(p[0], ax), proj(p[1], ax), proj(p[2], ax), proj(p[3], ax)];
        m.qi[i] = island_base + ax as i32 * 2 + sg;
    }
    for i in 0..m.t.len() {
        let f = m.t[i];
        let p: Vec<DVec3> = f.iter().map(|&k| m.v[k as usize]).collect();
        let n = (p[1] - p[0]).cross(p[2] - p[0]);
        let (ax, sg) = pick(n);
        m.tuv[i] = [proj(p[0], ax), proj(p[1], ax), proj(p[2], ax)];
        m.ti[i] = island_base + ax as i32 * 2 + sg;
    }
}

/// Catmull-Rom densify a polyline (n samples per segment).
pub fn catmull(p: &[DVec3], n: usize, closed: bool) -> Vec<DVec3> {
    let k = p.len();
    let q: Vec<DVec3> = if closed {
        let mut q = vec![p[k - 1]];
        q.extend_from_slice(p);
        q.push(p[0]);
        q.push(p[1]);
        q
    } else {
        let mut q = vec![p[0] * 2.0 - p[1]];
        q.extend_from_slice(p);
        q.push(p[k - 1] * 2.0 - p[k - 2]);
        q
    };
    let segs = if closed { k } else { k - 1 };
    let mut out = Vec::new();
    for i in 0..segs {
        let (p0, p1, p2, p3) = (q[i], q[i + 1], q[i + 2], q[i + 3]);
        for s in 0..n {
            let t = s as f64 / n as f64;
            out.push(
                (p1 * 2.0 + (p2 - p0) * t + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (t * t) + (-p0 + p1 * 3.0 - p2 * 3.0 + p3) * (t * t * t))
                    * 0.5,
            );
        }
    }
    if !closed {
        out.push(p[k - 1]);
    }
    out
}

/// Catmull-Rom on 2D points (curves in a chart).
pub fn catmull2(p: &[DVec2], n: usize, closed: bool) -> Vec<DVec2> {
    let p3: Vec<DVec3> = p.iter().map(|q| DVec3::new(q.x, q.y, 0.0)).collect();
    catmull(&p3, n, closed).into_iter().map(|q| DVec2::new(q.x, q.y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbox_is_closed_and_outward() {
        let m = rbox(DVec3::ZERO, DVec3::new(0.2, 0.1, 0.3), 0.03, 2, Mat::Chrome);
        let mut vol = 0.0;
        for f in m.triangles() {
            let (a, b, c) = (m.v[f[0] as usize], m.v[f[1] as usize], m.v[f[2] as usize]);
            vol += a.dot(b.cross(c)) / 6.0;
        }
        // a rounded box is a little under 0.4 x 0.2 x 0.6
        assert!(vol > 0.040 && vol < 0.048, "{vol}");
        let (lo, hi) = m.bbox();
        assert!((hi - lo - DVec3::new(0.4, 0.2, 0.6)).abs().max_element() < 1e-9);
    }

    #[test]
    fn rotations_match_python() {
        // rot_x(a) = [[1,0,0],[0,c,-s],[0,s,c]] (row-major)
        let r = rot_x(0.3);
        let v = r * DVec3::new(0.0, 1.0, 0.0);
        assert!((v - DVec3::new(0.0, 0.3f64.cos(), 0.3f64.sin())).length() < 1e-12);
        let r = rot_y(0.3);
        let v = r * DVec3::new(1.0, 0.0, 0.0);
        assert!((v - DVec3::new(0.3f64.cos(), 0.0, -(0.3f64.sin()))).length() < 1e-12);
    }
}
