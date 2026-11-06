//! The generated sedan's render model, to the vehicle art spec (the glTF
//! delivery an artist would send): every part a named mesh, the spec's
//! materials, markers for wheels, hinges, latches, bolts and the H-points.
//!
//! Panels are cut from one smooth body surface ([`super::surface`]) as
//! strips: a strip runs along one ray coordinate `a` (from `a0(t)` to
//! `a1(t)`) and across the other, from `lo(a)` to `hi(a)` at t = 0 … 1. Each
//! strip is a quad grid evened out to about [`H`] in 3D, its normals the
//! surface's gradient. Panel edges on shut lines step back [`GAP`] (a 4 mm
//! gap) and roll under in a hem. Every strip keeps its (s, t) as a
//! parametric (u, v), so a later binding can be built by construction.

use super::surface::{
    self as sf, Param, ARCH_R, AXLE_F, AXLE_R, RIM_R, TRACK_HALF, TYRE_R, TYRE_W, ZC_F, ZC_R,
};
use glam::{Mat3, Quat, Vec3};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

/// Target edge length on panels (m): the spec's 15–30 mm.
pub const H: f32 = 0.017;
/// Half a shut-line gap (m): 4 mm between parts.
pub const GAP: f32 = 0.002;
/// Hem: the skin rolls over 6.5 mm (a return wall [`HEM_R`] × 2 deep), its
/// foot tucked [`HEM_FLANGE`] back under the skin. One row deep, so its
/// quads stay within the 6:1 aspect at the panels' edge spacing, and the
/// shut line measures the same at the foot as at the face.
pub const HEM_R: f32 = 0.00325;
pub const HEM_FLANGE: f32 = 0.0005;

/// The spec's materials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mat {
    Paint,
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

impl Mat {
    pub const ALL: [Mat; 10] = [
        Mat::Paint,
        Mat::Chrome,
        Mat::TrimBlack,
        Mat::Rubber,
        Mat::Glass,
        Mat::LampLens,
        Mat::LampEmissive,
        Mat::InteriorFabric,
        Mat::InteriorPlastic,
        Mat::MetalBare,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Mat::Paint => "paint",
            Mat::Chrome => "chrome",
            Mat::TrimBlack => "trim_black",
            Mat::Rubber => "rubber",
            Mat::Glass => "glass",
            Mat::LampLens => "lamp_lens",
            Mat::LampEmissive => "lamp_emissive",
            Mat::InteriorFabric => "interior_fabric",
            Mat::InteriorPlastic => "interior_plastic",
            Mat::MetalBare => "metal_bare",
        }
    }
    /// glTF PBR: base colour (linear RGBA), metallic, roughness.
    pub fn pbr(self) -> ([f32; 4], f32, f32) {
        match self {
            // A deep metallic blue (the base coat; the clearcoat is an extension).
            Mat::Paint => ([0.016, 0.040, 0.120, 1.0], 0.6, 0.35),
            Mat::Chrome => ([0.92, 0.92, 0.93, 1.0], 1.0, 0.08),
            Mat::TrimBlack => ([0.022, 0.022, 0.025, 1.0], 0.0, 0.55),
            Mat::Rubber => ([0.025, 0.025, 0.025, 1.0], 0.0, 0.85),
            Mat::Glass => ([0.06, 0.07, 0.08, 0.35], 0.0, 0.03),
            Mat::LampLens => ([0.95, 0.95, 0.97, 0.25], 0.0, 0.03),
            Mat::LampEmissive => ([1.0, 0.97, 0.92, 1.0], 0.0, 0.3),
            Mat::InteriorFabric => ([0.07, 0.07, 0.075, 1.0], 0.0, 0.9),
            Mat::InteriorPlastic => ([0.05, 0.05, 0.055, 1.0], 0.0, 0.6),
            Mat::MetalBare => ([0.55, 0.56, 0.58, 1.0], 1.0, 0.35),
        }
    }
}

/// A UV chart: a strip's vertices with their local (u, v) in metres.
#[derive(Clone, Debug)]
pub struct Chart {
    pub first: u32,
    pub count: u32,
    pub size: [f32; 2],
}

/// One part (a glTF node with its mesh).
#[derive(Clone, Debug, Default)]
pub struct Part {
    pub name: String,
    pub pos: Vec<Vec3>,
    pub nrm: Vec<Vec3>,
    pub uv: Vec<[f32; 2]>,
    /// Per vertex colour (linear RGB): white except where a shared material
    /// is tinted (a tail lamp's red lens).
    pub col: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
    pub tmat: Vec<Mat>,
    pub charts: Vec<Chart>,
}

impl Part {
    fn new(name: &str) -> Self {
        Part {
            name: name.into(),
            ..Default::default()
        }
    }
    fn vert(&mut self, p: Vec3, n: Vec3, uv: [f32; 2], col: [f32; 3]) -> u32 {
        self.pos.push(p);
        self.nrm.push(n);
        self.uv.push(uv);
        self.col.push(col);
        (self.pos.len() - 1) as u32
    }
    /// A triangle facing its vertices' normals.
    fn tri(&mut self, t: [u32; 3], m: Mat) {
        let [a, b, c] = t.map(|i| self.pos[i as usize]);
        let g = (b - a).cross(c - a);
        if g.length_squared() < 1e-14 {
            return;
        }
        // Three nearly collinear points (a grid folding along a kinked
        // border) span no surface: drop the sliver.
        let l = (b - a)
            .length_squared()
            .max((c - b).length_squared())
            .max((a - c).length_squared());
        if l / g.length() > 40.0 {
            return;
        }
        let n = self.nrm[t[0] as usize] + self.nrm[t[1] as usize] + self.nrm[t[2] as usize];
        self.tris.push(if g.dot(n) >= 0.0 {
            t
        } else {
            [t[0], t[2], t[1]]
        });
        self.tmat.push(m);
    }
    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32, m: Mat) {
        // Split along the shorter diagonal.
        let p = |i: u32| self.pos[i as usize];
        if p(a).distance_squared(p(c)) <= p(b).distance_squared(p(d)) {
            self.tri([a, b, c], m);
            self.tri([a, c, d], m);
        } else {
            self.tri([a, b, d], m);
            self.tri([b, c, d], m);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Marker {
    pub name: String,
    pub pos: Vec3,
    pub rot: Quat,
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub parts: Vec<Part>,
    pub markers: Vec<Marker>,
}

impl Model {
    fn part(&mut self, name: &str) -> &mut Part {
        if let Some(k) = self.parts.iter().position(|p| p.name == name) {
            return &mut self.parts[k];
        }
        self.parts.push(Part::new(name));
        self.parts.last_mut().unwrap()
    }
    fn marker(&mut self, name: &str, pos: Vec3) {
        self.markers.push(Marker {
            name: name.into(),
            pos,
            rot: Quat::IDENTITY,
        });
    }
}

// ---------------------------------------------------------------------------
// Strips over the surface.
// ---------------------------------------------------------------------------

/// A function of one variable tabulated (the boundary functions call the
/// surface's root finders, which are slow).
#[derive(Clone)]
pub struct Table {
    lo: f32,
    hi: f32,
    v: Vec<f32>,
}

impl Table {
    pub fn new(f: impl Fn(f32) -> f32, lo: f32, hi: f32) -> Self {
        let n = 240;
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        let v = (0..=n)
            .map(|k| f(lo + (hi - lo) * k as f32 / n as f32))
            .collect();
        Table { lo, hi, v }
    }
    pub fn at(&self, x: f32) -> f32 {
        let n = self.v.len() - 1;
        if self.hi <= self.lo {
            return self.v[0];
        }
        let f = ((x - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0) * n as f32;
        let i = (f.floor() as usize).min(n - 1);
        let w = f - i as f32;
        self.v[i] * (1.0 - w) + self.v[i + 1] * w
    }
}

type F = Box<dyn Fn(f32) -> f32>;

/// A strip: a from a0(t) to a1(t) (s = 0 … 1), b from lo(a) to hi(a)
/// (t = 0 … 1), on `param`; `off` moves it along the surface normal.
pub struct Strip {
    pub param: Param,
    pub a0: F,
    pub a1: F,
    pub lo: F,
    pub hi: F,
    pub off: f32,
    /// Target edge length (m).
    pub h: f32,
    /// Character lines across the strip (each its b as a function of a, in
    /// order of increasing t): a grid row runs exactly along each (an edge
    /// loop on the crease).
    pub creases: Vec<F>,
}

fn k(v: f32) -> F {
    Box::new(move |_| v)
}

impl Strip {
    pub fn new(param: Param, a0: F, a1: F, lo: F, hi: F) -> Self {
        Strip {
            param,
            a0,
            a1,
            lo,
            hi,
            off: 0.0,
            h: H,
            creases: Vec::new(),
        }
    }
    /// The t of crease `c` at s.
    fn crease_t(&self, c: usize, s: f32) -> f32 {
        let f = &self.creases[c];
        let mut t = 0.5;
        for _ in 0..8 {
            let a = (self.a0)(t) + ((self.a1)(t) - (self.a0)(t)) * s;
            let (lo, hi) = ((self.lo)(a), (self.hi)(a));
            t = ((f(a) - lo) / (hi - lo)).clamp(0.06, 0.94);
        }
        t
    }
    fn ab(&self, s: f32, t: f32) -> (f32, f32) {
        let a = (self.a0)(t) + ((self.a1)(t) - (self.a0)(t)) * s;
        let b = (self.lo)(a) + t * ((self.hi)(a) - (self.lo)(a));
        (a, b)
    }
    pub fn point(&self, s: f32, t: f32) -> Vec3 {
        let (a, b) = self.ab(s, t);
        sf::hit(self.param, a, b)
    }
}

/// A quad grid: (ns + 1) × (nb + 1) points, row-major in t.
#[derive(Clone, Debug)]
pub struct Grid {
    pub ns: usize,
    pub nb: usize,
    pub p: Vec<Vec3>,
    pub n: Vec<Vec3>,
    /// Local (u, v) in metres.
    pub uv: Vec<[f32; 2]>,
}

impl Grid {
    pub fn at(&self, i: usize, j: usize) -> usize {
        j * (self.ns + 1) + i
    }
    pub fn mirrored(&self) -> Grid {
        let f = |v: &Vec3| Vec3::new(-v.x, v.y, v.z);
        Grid {
            ns: self.ns,
            nb: self.nb,
            p: self.p.iter().map(f).collect(),
            n: self.n.iter().map(f).collect(),
            uv: self.uv.clone(),
        }
    }
    pub fn offset(&self, d: f32) -> Grid {
        let mut g = self.clone();
        for (p, n) in g.p.iter_mut().zip(&g.n) {
            *p += *n * d;
        }
        g
    }
    pub fn flipped(&self) -> Grid {
        let mut g = self.clone();
        for n in &mut g.n {
            *n = -*n;
        }
        g
    }
    /// The edge's indices in order: 0 lo (t = 0), 1 hi, 2 start (s = 0), 3 end.
    pub fn edge(&self, e: usize) -> Vec<usize> {
        match e {
            0 => (0..=self.ns).map(|i| self.at(i, 0)).collect(),
            1 => (0..=self.ns).map(|i| self.at(i, self.nb)).collect(),
            2 => (0..=self.nb).map(|j| self.at(0, j)).collect(),
            _ => (0..=self.nb).map(|j| self.at(self.ns, j)).collect(),
        }
    }
    pub fn edge_points(&self, e: usize) -> Vec<Vec3> {
        self.edge(e).iter().map(|&k| self.p[k]).collect()
    }
    /// Step the flagged edges back along the surface by `g` (shut lines).
    pub fn shrink(&mut self, edges: [bool; 4], g: f32) {
        self.shrink_by(edges.map(|e| if e { g } else { 0.0 }));
    }
    /// Step each edge back along the surface by its own distance (a corner
    /// moves along both edges' inward directions).
    pub fn shrink_by(&mut self, d: [f32; 4]) {
        let (ns, nb) = (self.ns, self.nb);
        let orig = self.p.clone();
        for j in 0..=nb {
            for i in 0..=ns {
                let k = self.at(i, j);
                let n = self.n[k];
                // Inward across the edge: toward the next row or column, less
                // its part along the edge (sheared grids), in the tangent plane.
                let inward = |inner: usize, a: usize, b: usize| {
                    let along = (orig[b] - orig[a]).normalize_or_zero();
                    let x = along.cross(n).normalize_or_zero();
                    if x.dot(orig[inner] - orig[k]) >= 0.0 {
                        x
                    } else {
                        -x
                    }
                };
                let (ip, inx) = (i.saturating_sub(1), (i + 1).min(ns));
                let (jp, jn) = (j.saturating_sub(1), (j + 1).min(nb));
                let mut mv = Vec3::ZERO;
                if d[0] > 0.0 && j == 0 {
                    mv += inward(self.at(i, 1.min(nb)), self.at(ip, j), self.at(inx, j)) * d[0];
                }
                if d[1] > 0.0 && j == nb {
                    mv += inward(
                        self.at(i, nb.saturating_sub(1)),
                        self.at(ip, j),
                        self.at(inx, j),
                    ) * d[1];
                }
                if d[2] > 0.0 && i == 0 {
                    mv += inward(self.at(1.min(ns), j), self.at(i, jp), self.at(i, jn)) * d[2];
                }
                if d[3] > 0.0 && i == ns {
                    mv += inward(
                        self.at(ns.saturating_sub(1), j),
                        self.at(i, jp),
                        self.at(i, jn),
                    ) * d[3];
                }
                self.p[k] += mv;
            }
        }
    }
}

fn polylen(p: &[Vec3]) -> f32 {
    p.windows(2).map(|w| w[0].distance(w[1])).sum()
}

/// Positions along a polyline at even arclength fractions of `n` steps,
/// given as (parameter pairs, points).
fn even(params: &[(f32, f32)], pts: &[Vec3], n: usize) -> Vec<(f32, f32)> {
    let mut cum = vec![0.0f32; pts.len()];
    for k in 1..pts.len() {
        cum[k] = cum[k - 1] + pts[k].distance(pts[k - 1]);
    }
    let total = *cum.last().unwrap();
    (0..=n)
        .map(|i| {
            let target = total * i as f32 / n as f32;
            let k = cum.partition_point(|&c| c < target).clamp(1, pts.len() - 1);
            let seg = (cum[k] - cum[k - 1]).max(1e-9);
            let w = ((target - cum[k - 1]) / seg).clamp(0.0, 1.0);
            let (a, b) = (params[k - 1], params[k]);
            (a.0 + (b.0 - a.0) * w, a.1 + (b.1 - a.1) * w)
        })
        .collect()
}

/// The strip as a grid evened out to about `h`.
pub fn grid(st: &Strip, h: f32) -> Grid {
    let line = |f: &dyn Fn(f32) -> (f32, f32)| -> Vec<Vec3> {
        (0..=48)
            .map(|k| {
                let (s, t) = f(k as f32 / 48.0);
                st.point(s, t)
            })
            .collect()
    };
    let l_lo = polylen(&line(&|x| (x, 0.0)));
    let l_hi = polylen(&line(&|x| (x, 1.0)));
    let l_st = polylen(&line(&|x| (0.0, x)));
    let l_en = polylen(&line(&|x| (1.0, x)));
    let ns = ((l_lo.max(l_hi) / h).round() as usize).max(1);
    let nb = ((l_st.max(l_en) / h).round() as usize).max(1);
    // Crease rows: each pinned to its crease; the columns even out between
    // them.
    let nc = st.creases.len();
    let nb = nb.max(nc + 1);
    let mut jcs: Vec<usize> = Vec::new();
    for c in 0..nc {
        let mean = (0..=8).map(|q| st.crease_t(c, q as f32 / 8.0)).sum::<f32>() / 9.0;
        let lo = jcs.last().map_or(1, |&j| j + 1);
        let j = ((mean * nb as f32).round() as usize).clamp(lo, nb - (nc - c));
        jcs.push(j);
    }
    // Pins (row, t at s): the borders and the crease rows.
    let pins = |s: f32| -> Vec<(usize, f32)> {
        let mut v = vec![(0usize, 0.0f32)];
        for (c, &j) in jcs.iter().enumerate() {
            v.push((j, st.crease_t(c, s)));
        }
        v.push((nb, 1.0));
        v
    };
    let mut st_ij: Vec<(f32, f32)> = (0..=nb)
        .flat_map(|j| {
            let pins = &pins;
            (0..=ns).map(move |i| {
                let s = i as f32 / ns as f32;
                let pv = pins(s);
                let k = pv.partition_point(|p| p.0 <= j).clamp(1, pv.len() - 1);
                let (a, b) = (pv[k - 1], pv[k]);
                let t = a.1 + (b.1 - a.1) * (j - a.0) as f32 / (b.0 - a.0).max(1) as f32;
                (s, t)
            })
        })
        .collect();
    let at = |i: usize, j: usize| j * (ns + 1) + i;
    let sub = 3;
    for _ in 0..3 {
        // Rows: even in arclength along s.
        for j in 0..=nb {
            let mut prm = Vec::new();
            for i in 0..ns {
                let (a, b) = (st_ij[at(i, j)], st_ij[at(i + 1, j)]);
                for q in 0..sub {
                    let w = q as f32 / sub as f32;
                    prm.push((a.0 + (b.0 - a.0) * w, a.1 + (b.1 - a.1) * w));
                }
            }
            prm.push(st_ij[at(ns, j)]);
            let pts: Vec<Vec3> = prm.iter().map(|&(s, t)| st.point(s, t)).collect();
            let e = even(&prm, &pts, ns);
            for i in 1..ns {
                st_ij[at(i, j)] = e[i];
                if let Some(c) = jcs.iter().position(|&jc| jc == j) {
                    st_ij[at(i, j)].1 = st.crease_t(c, e[i].0);
                }
            }
        }
        // Columns (in runs between the crease rows).
        let mut bounds = vec![0usize];
        bounds.extend(jcs.iter().copied());
        bounds.push(nb);
        let runs: Vec<(usize, usize)> = bounds.windows(2).map(|w| (w[0], w[1])).collect();
        for i in 0..=ns {
            for &(j0, j1) in &runs {
                let mut prm = Vec::new();
                for j in j0..j1 {
                    let (a, b) = (st_ij[at(i, j)], st_ij[at(i, j + 1)]);
                    for q in 0..sub {
                        let w = q as f32 / sub as f32;
                        prm.push((a.0 + (b.0 - a.0) * w, a.1 + (b.1 - a.1) * w));
                    }
                }
                prm.push(st_ij[at(i, j1)]);
                let pts: Vec<Vec3> = prm.iter().map(|&(s, t)| st.point(s, t)).collect();
                let e = even(&prm, &pts, j1 - j0);
                for j in j0 + 1..j1 {
                    st_ij[at(i, j)] = e[j - j0];
                }
            }
        }
    }
    let p: Vec<Vec3> = st_ij.iter().map(|&(s, t)| st.point(s, t)).collect();
    let n: Vec<Vec3> = p.iter().map(|&q| sf::normal(q)).collect();
    let p: Vec<Vec3> = p.iter().zip(&n).map(|(q, n)| *q + *n * st.off).collect();
    // Local UV: the grid's (i, j) lattice scaled by its mean row and column
    // lengths (rows and columns are even in arclength, so this keeps the
    // texel density about even and can never fold).
    let row_len = (0..=nb)
        .map(|j| {
            (0..ns)
                .map(|i| p[at(i, j)].distance(p[at(i + 1, j)]))
                .sum::<f32>()
        })
        .sum::<f32>()
        / (nb + 1) as f32;
    let col_len = (0..=ns)
        .map(|i| {
            (0..nb)
                .map(|j| p[at(i, j)].distance(p[at(i, j + 1)]))
                .sum::<f32>()
        })
        .sum::<f32>()
        / (ns + 1) as f32;
    let uv = (0..=nb)
        .flat_map(|j| {
            (0..=ns).map(move |i| {
                [
                    row_len * i as f32 / ns as f32,
                    col_len * j as f32 / nb as f32,
                ]
            })
        })
        .collect();
    Grid { ns, nb, p, n, uv }
}

/// A triangle's aspect (longest edge squared over twice its area: 1.15 for
/// an equilateral triangle, the spec's limit 6).
fn aspect(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let area = 0.5 * (b - a).cross(c - a).length();
    let l = (b - a)
        .length_squared()
        .max((c - b).length_squared())
        .max((a - c).length_squared());
    l / (2.0 * area.max(1e-14))
}

/// Whether any of `new` (vertex triples) overlaps, in UV, a live triangle of
/// `part.tris[tri0..]` other than those in `skip` (interiors only: sharing
/// an edge or a corner is not an overlap).
fn uv_overlaps(part: &Part, tri0: usize, dead: &[bool], new: &[[u32; 3]], skip: &[usize]) -> bool {
    let uvt = |t: [u32; 3]| t.map(|x| part.uv[x as usize]);
    let sep = |a: [[f32; 2]; 3], b: [[f32; 2]; 3]| -> bool {
        for tri in [a, b] {
            for e in 0..3 {
                let (p, q) = (tri[e], tri[(e + 1) % 3]);
                let n = [q[1] - p[1], p[0] - q[0]];
                let l = (n[0] * n[0] + n[1] * n[1]).sqrt();
                if l < 1e-12 {
                    continue;
                }
                let pr = |v: [f32; 2]| (v[0] * n[0] + v[1] * n[1]) / l;
                let (mut a0, mut a1, mut b0, mut b1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for v in a {
                    a0 = a0.min(pr(v));
                    a1 = a1.max(pr(v));
                }
                for v in b {
                    b0 = b0.min(pr(v));
                    b1 = b1.max(pr(v));
                }
                let eps = 2e-5;
                if a1 <= b0 + eps || b1 <= a0 + eps {
                    return true;
                }
            }
        }
        false
    };
    for nt in new {
        let a = uvt(*nt);
        let (lo, hi) = a.iter().fold(([f32::MAX; 2], [f32::MIN; 2]), |(l, h), v| {
            (
                [l[0].min(v[0]), l[1].min(v[1])],
                [h[0].max(v[0]), h[1].max(v[1])],
            )
        });
        for r in tri0..part.tris.len() {
            if dead[r] || skip.contains(&r) {
                continue;
            }
            let b = uvt(part.tris[r]);
            if b.iter().all(|v| v[0] < lo[0]) || b.iter().all(|v| v[0] > hi[0]) {
                continue;
            }
            if b.iter().all(|v| v[1] < lo[1]) || b.iter().all(|v| v[1] > hi[1]) {
                continue;
            }
            if !sep(a, b) {
                return true;
            }
        }
    }
    false
}

/// Remove slivers from the triangles `part.tris[tri0..]` (one strip, after
/// its quads are split): a thin triangle with a short edge has that edge
/// collapsed (into the end on the strip's border, which never moves, so the
/// outline and its shut line stay put), and a flat one with no short edge
/// has its longest edge flipped where that gives the pair a better worst
/// shape. Vertices only ever merge into one another: every point stays on
/// the surface. `ids` (grid point → vertex) follows the merges.
fn clean_slivers(part: &mut Part, tri0: usize, first: usize, border: &[bool], ids: &mut [u32]) {
    use std::collections::HashMap;
    const LIMIT: f32 = 4.5;
    let on_border = |v: u32| -> bool {
        let k = (v as usize).wrapping_sub(first);
        k < border.len() && border[k]
    };
    let mut merged: HashMap<u32, u32> = HashMap::new();
    for _pass in 0..30 {
        let mut changed = false;
        // Edge → the triangles on it.
        let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for t in tri0..part.tris.len() {
            let tr = part.tris[t];
            for e in 0..3 {
                let (a, b) = (tr[e], tr[(e + 1) % 3]);
                edges.entry((a.min(b), a.max(b))).or_default().push(t);
            }
        }
        let mut dead = vec![false; part.tris.len()];
        let mut touched = vec![false; part.tris.len()];
        for t in tri0..part.tris.len() {
            if dead[t] || touched[t] {
                continue;
            }
            let tr = part.tris[t];
            let p = tr.map(|v| part.pos[v as usize]);
            if aspect(p[0], p[1], p[2]) <= LIMIT {
                continue;
            }
            let len = |e: usize| p[e].distance(p[(e + 1) % 3]);
            let es = (0..3).min_by(|&x, &y| len(x).total_cmp(&len(y))).unwrap();
            let el = (0..3).max_by(|&x, &y| len(x).total_cmp(&len(y))).unwrap();
            // A sliver of three points along the strip's outline (a kinked
            // border) spans no visible surface: drop it.
            let height = (p[1] - p[0]).cross(p[2] - p[0]).length() / len(el);
            let on_outline = (0..3).any(|e| {
                let (a, b) = (tr[e], tr[(e + 1) % 3]);
                edges.get(&(a.min(b), a.max(b))).map_or(0, |v| v.len()) == 1
            });
            if (tr.iter().all(|&v| on_border(v)) && height < 0.0035)
                || (on_outline && height < 0.002)
            {
                dead[t] = true;
                changed = true;
                continue;
            }
            // Candidate collapses (drop `v` into `k`): a needle's short edge
            // either way; a thin cap's apex onto either end of its longest
            // edge. A border point only moves along the border, a short step
            // (the hem's spacing stays under 22 mm).
            let mut cands: Vec<(u32, u32)> = Vec::new();
            let border_ok = |a: u32, b: u32, l: f32| {
                let key = (a.min(b), a.max(b));
                let boundary_edge = edges.get(&key).map_or(0, |v| v.len()) == 1;
                !on_border(a) || (on_border(b) && boundary_edge && l < 0.45 * H)
            };
            if len(es) < 0.5 * len(el) {
                let (a, b) = (tr[es], tr[(es + 1) % 3]);
                for (v, k) in [(a, b), (b, a)] {
                    if border_ok(v, k, len(es)) {
                        cands.push((v, k));
                    }
                }
            } else if height < 0.0025 {
                let (a, b, c) = (tr[el], tr[(el + 1) % 3], tr[(el + 2) % 3]);
                let (pa, pb, pc) = (
                    part.pos[a as usize],
                    part.pos[b as usize],
                    part.pos[c as usize],
                );
                let mut ends = [(a, pc.distance(pa)), (b, pc.distance(pb))];
                ends.sort_by(|x, y| x.1.total_cmp(&y.1));
                for (k, l) in ends {
                    if border_ok(c, k, l) {
                        cands.push((c, k));
                    }
                }
            }
            let mut done = false;
            for (v, k) in cands {
                // The triangles round `v` must keep their facing.
                let around: Vec<usize> = (tri0..part.tris.len())
                    .filter(|&q| !dead[q] && part.tris[q].contains(&v))
                    .collect();
                // The link condition: the vertices next to both `v` and `k`
                // are just the apexes of the triangles on edge v–k (else the
                // collapse pinches the surface into a non-manifold edge).
                let ring = |w: u32| -> Vec<u32> {
                    let mut r: Vec<u32> = (tri0..part.tris.len())
                        .filter(|&q| !dead[q] && part.tris[q].contains(&w))
                        .flat_map(|q| part.tris[q])
                        .filter(|&x| x != w)
                        .collect();
                    r.sort_unstable();
                    r.dedup();
                    r
                };
                let (rv, rk) = (ring(v), ring(k));
                let shared = rv
                    .iter()
                    .filter(|x| rk.contains(x) && **x != k && **x != v)
                    .count();
                let apexes = around
                    .iter()
                    .filter(|&&q| part.tris[q].contains(&k))
                    .count();
                if shared != apexes {
                    continue;
                }
                let mut ok = true;
                for &q in &around {
                    let tq = part.tris[q];
                    if tq.contains(&k) {
                        continue;
                    }
                    let old = tq.map(|x| part.pos[x as usize]);
                    let new = tq.map(|x| part.pos[if x == v { k } else { x } as usize]);
                    let n0 = (old[1] - old[0]).cross(old[2] - old[0]);
                    let n1 = (new[1] - new[0]).cross(new[2] - new[0]);
                    if n0.dot(n1) <= 0.0 || aspect(new[0], new[1], new[2]) > 12.0 {
                        ok = false;
                        break;
                    }
                    // Nor may its UVs fold (the chart stays one-to-one).
                    let uv_area = |t: [u32; 3]| {
                        let [a, b, c] = t.map(|x| part.uv[x as usize]);
                        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    };
                    let (u0, u1) = (uv_area(tq), uv_area(tq.map(|x| if x == v { k } else { x })));
                    if u0 * u1 <= 0.0 {
                        ok = false;
                        break;
                    }
                    // No duplicate of a triangle already on `k`.
                    let mut s = tq.map(|x| if x == v { k } else { x });
                    s.sort_unstable();
                    let dup = (tri0..part.tris.len()).any(|r| {
                        let mut o = part.tris[r];
                        o.sort_unstable();
                        !dead[r] && r != q && o == s
                    });
                    if dup {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                // Nor may the moved triangles lap over others in the chart.
                let moved: Vec<[u32; 3]> = around
                    .iter()
                    .filter(|&&q| !part.tris[q].contains(&k))
                    .map(|&q| part.tris[q].map(|x| if x == v { k } else { x }))
                    .collect();
                if uv_overlaps(part, tri0, &dead, &moved, &around) {
                    continue;
                }
                for &q in &around {
                    let tq = part.tris[q];
                    if tq.contains(&k) {
                        dead[q] = true;
                    } else {
                        part.tris[q] = tq.map(|x| if x == v { k } else { x });
                    }
                    touched[q] = true;
                }
                merged.insert(v, k);
                changed = true;
                done = true;
                break;
            }
            if done {
                continue;
            }
            {
                // A cap thinner than half a millimetre across its longest
                // edge covers nothing visible: drop it.
                let area = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length();
                if 2.0 * area / len(el) < 0.0005 {
                    dead[t] = true;
                    changed = true;
                    continue;
                }
                // Flip the longest edge.
                let (a, b) = (tr[el], tr[(el + 1) % 3]);
                let c = tr[(el + 2) % 3];
                let Some(list) = edges.get(&(a.min(b), a.max(b))) else {
                    continue;
                };
                let Some(&t2) = list.iter().find(|&&q| q != t && !dead[q] && !touched[q]) else {
                    continue;
                };
                if list.len() != 2 || part.tmat[t2] != part.tmat[t] {
                    continue;
                }
                let o = part.tris[t2];
                let Some(&d) = o.iter().find(|&&x| x != a && x != b) else {
                    continue;
                };
                // The new edge must not already exist (a third triangle on it).
                if edges.contains_key(&(c.min(d), c.max(d))) {
                    continue;
                }
                let q = |x: u32| part.pos[x as usize];
                let before = aspect(q(a), q(b), q(c)).max(aspect(q(o[0]), q(o[1]), q(o[2])));
                // The new pair (c, a, d) and (c, d, b), wound like the old.
                let n_old = (q(tr[1]) - q(tr[0])).cross(q(tr[2]) - q(tr[0]));
                let t_new = [[c, a, d], [c, d, b]];
                let mut ok = true;
                let mut after = 0.0f32;
                let mut out = [[0u32; 3]; 2];
                for (s, nt) in t_new.iter().enumerate() {
                    let n = (q(nt[1]) - q(nt[0])).cross(q(nt[2]) - q(nt[0]));
                    let nt = if n.dot(n_old) >= 0.0 {
                        *nt
                    } else {
                        [nt[0], nt[2], nt[1]]
                    };
                    let n = (q(nt[1]) - q(nt[0])).cross(q(nt[2]) - q(nt[0]));
                    // Convex: both new triangles face the old way.
                    if n.dot(n_old) <= 0.0 || n.length_squared() < 1e-14 {
                        ok = false;
                    }
                    after = after.max(aspect(q(nt[0]), q(nt[1]), q(nt[2])));
                    out[s] = nt;
                }
                // The quad must be convex: c and d on opposite sides of a-b
                // and a, b on opposite sides of c-d.
                let side = |p0: Vec3, p1: Vec3, x: Vec3| (p1 - p0).cross(x - p0).dot(n_old);
                if side(q(c), q(d), q(a)) * side(q(c), q(d), q(b)) >= 0.0 {
                    ok = false;
                }
                // Nor may the UVs fold (both new triangles wound as the old).
                let uv_area = |t: [u32; 3]| {
                    let [a, b, c] = t.map(|x| part.uv[x as usize]);
                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                };
                let s0 = uv_area(tr);
                if out.iter().any(|&nt| uv_area(nt) * s0 <= 0.0) {
                    ok = false;
                }
                if !ok || after >= before * 0.95 {
                    continue;
                }
                if uv_overlaps(part, tri0, &dead, &out, &[t, t2]) {
                    continue;
                }
                part.tris[t] = out[0];
                part.tris[t2] = out[1];
                touched[t] = true;
                touched[t2] = true;
                changed = true;
            }
        }
        // Drop the dead triangles.
        let mut w = tri0;
        for t in tri0..part.tris.len() {
            if !dead[t] {
                part.tris[w] = part.tris[t];
                part.tmat[w] = part.tmat[t];
                w += 1;
            }
        }
        part.tris.truncate(w);
        part.tmat.truncate(w);
        if !changed {
            break;
        }
    }
    for id in ids.iter_mut() {
        let mut v = *id;
        while let Some(&k) = merged.get(&v) {
            v = k;
        }
        *id = v;
    }
}

/// Append a grid to a part (one UV chart); returns its vertex indices.
pub fn emit(part: &mut Part, g: &Grid, m: Mat, col: [f32; 3]) -> Vec<u32> {
    let first = part.pos.len() as u32;
    let tri0 = part.tris.len();
    let raw: Vec<u32> = (0..g.p.len())
        .map(|k| part.vert(g.p[k], g.n[k], g.uv[k], col))
        .collect();
    // Where a strip tapers its rows (or columns) crowd together: collapse
    // grid points closer than a fifth of an edge onto their neighbour along
    // the column (then the row), toward the border, so a narrow end is
    // triangulated with well-shaped triangles rather than slivers.
    let thr = 0.45 * H;
    let mut rep: Vec<usize> = (0..g.p.len()).collect();
    fn root(rep: &mut [usize], k: usize) -> usize {
        let mut r = k;
        while rep[r] != r {
            r = rep[r];
        }
        rep[k] = r;
        r
    }
    let lines: Vec<Vec<usize>> = (0..=g.ns)
        .map(|i| (0..=g.nb).map(|j| g.at(i, j)).collect())
        .chain((0..=g.nb).map(|j| (0..=g.ns).map(|i| g.at(i, j)).collect()))
        .collect();
    for line in &lines {
        let n = line.len();
        if n < 3 {
            continue;
        }
        // From each end inward, merging into the kept point (ends are kept).
        let mut kept = line[0];
        for q in 1..n - 1 {
            let (a, b) = (root(&mut rep, line[q]), root(&mut rep, kept));
            if g.p[a].distance(g.p[b]) < thr && a != b {
                rep[a] = b;
            } else {
                kept = line[q];
            }
        }
        let last = line[n - 1];
        let (a, b) = (root(&mut rep, kept), root(&mut rep, last));
        if kept != line[0] && a != b && g.p[a].distance(g.p[b]) < thr {
            rep[a] = b;
        }
    }
    let mut ids: Vec<u32> = (0..g.p.len()).map(|k| raw[root(&mut rep, k)]).collect();
    for j in 0..g.nb {
        for i in 0..g.ns {
            let q = [
                ids[g.at(i, j)],
                ids[g.at(i + 1, j)],
                ids[g.at(i + 1, j + 1)],
                ids[g.at(i, j + 1)],
            ];
            let mut u: Vec<u32> = Vec::with_capacity(4);
            for v in q {
                if !u.contains(&v) {
                    u.push(v);
                }
            }
            match u.len() {
                4 => part.quad(q[0], q[1], q[2], q[3], m),
                3 => {
                    // A collapsed quad along a border can leave three nearly
                    // collinear border points: a sliver of no area, dropped.
                    let [a, b, c] = [u[0], u[1], u[2]].map(|v| part.pos[v as usize]);
                    let area = 0.5 * (b - a).cross(c - a).length();
                    let l = (b - a).length().max((c - b).length()).max((a - c).length());
                    if l * l / (2.0 * area.max(1e-12)) <= 12.0 {
                        part.tri([u[0], u[1], u[2]], m);
                    }
                }
                _ => {}
            }
        }
    }
    let border: Vec<bool> = (0..g.p.len())
        .map(|k| {
            let (i, j) = (k % (g.ns + 1), k / (g.ns + 1));
            i == 0 || j == 0 || i == g.ns || j == g.nb
        })
        .collect();
    // Merged points can fold a triangle over its neighbours in UV (it then
    // spans no surface of its own either): drop those against the chart's
    // winding.
    {
        let uv_area = |t: &[u32; 3]| {
            let [a, b, c] = t.map(|x| part.uv[x as usize]);
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        };
        let sum: f32 = part.tris[tri0..].iter().map(|t| uv_area(t).signum()).sum();
        let keep: Vec<bool> = part.tris[tri0..]
            .iter()
            .map(|t| uv_area(t) * sum > 0.0)
            .collect();
        let mut w = tri0;
        for (q, &k) in keep.iter().enumerate() {
            if k {
                part.tris[w] = part.tris[tri0 + q];
                part.tmat[w] = part.tmat[tri0 + q];
                w += 1;
            }
        }
        part.tris.truncate(w);
        part.tmat.truncate(w);
    }
    clean_slivers(part, tri0, first as usize, &border, &mut ids);
    let size =
        g.uv.iter()
            .fold([0.0f32; 2], |s, u| [s[0].max(u[0]), s[1].max(u[1])]);
    part.charts.push(Chart {
        first,
        count: g.p.len() as u32,
        size,
    });
    ids
}

/// Fold a grid edge under (a hem), inside the panel's outline so the shut
/// line keeps its width: the edge turns down through 90° (its vertices'
/// normals lean out 45°, so the edge shades round), a wall [`HEM_R`] × 2
/// deep, its foot tucked [`HEM_FLANGE`] in under the skin.
pub fn hem(part: &mut Part, g: &Grid, ids: &[u32], e: usize, m: Mat) {
    let all = g.edge(e);
    let inner_all: Vec<usize> = match e {
        0 => (0..=g.ns).map(|i| g.at(i, 1.min(g.nb))).collect(),
        1 => (0..=g.ns)
            .map(|i| g.at(i, g.nb.saturating_sub(1)))
            .collect(),
        2 => (0..=g.nb).map(|j| g.at(1.min(g.ns), j)).collect(),
        _ => (0..=g.nb)
            .map(|j| g.at(g.ns.saturating_sub(1), j))
            .collect(),
    };
    // The edge's distinct vertices (merged points once), at least 5 mm
    // apart (a crowded end would make the return wall's quads slivers).
    let (mut idx, mut inner) = (Vec::new(), Vec::new());
    for (q, &k) in all.iter().enumerate() {
        let last = q + 1 == all.len();
        if let Some(&pk) = idx.last() {
            let (a, b) = (ids[pk] as usize, ids[k] as usize);
            if a == b || (!last && part.pos[a].distance(part.pos[b]) < 0.009) {
                continue;
            }
        }
        idx.push(k);
        inner.push(inner_all[q]);
    }
    // Where two hemmed edges meet at a sharp corner their returns would
    // cross: the wall stops one point short of an acute corner.
    let acute = |end: usize, next: usize| {
        let k = all[end];
        let (a, b) = (g.p[all[next]] - g.p[k], g.p[inner_all[end]] - g.p[k]);
        a.normalize_or_zero().dot(b.normalize_or_zero()) > 0.5
    };
    if idx.len() > 3 && acute(0, 1) {
        idx.remove(0);
        inner.remove(0);
    }
    if idx.len() > 3 && acute(all.len() - 1, all.len() - 2) {
        idx.pop();
        inner.pop();
    }
    if idx.len() < 2 {
        return;
    }
    let mut prev: Vec<u32> = idx.iter().map(|&k| ids[k]).collect();
    let rows: Vec<Vec<(Vec3, Vec3)>> = {
        let mut rows = vec![Vec::new(); 1];
        for (q, &k) in idx.iter().enumerate() {
            let p = part.pos[ids[k] as usize];
            let n = g.n[k];
            let o = {
                // Across the edge in the surface: tangent × normal, away
                // from the panel.
                let along = (g.p[idx[(q + 1).min(idx.len() - 1)]] - g.p[idx[q.saturating_sub(1)]])
                    .normalize_or_zero();
                let x = along.cross(n).normalize_or(Vec3::X);
                if x.dot(p - g.p[inner[q]]) >= 0.0 {
                    x
                } else {
                    -x
                }
            };
            let v = ids[k] as usize;
            part.nrm[v] = (part.nrm[v].normalize_or(n) + o).normalize_or(n);
            rows[0].push((
                p - n * (2.0 * HEM_R) - o * HEM_FLANGE,
                (o - n * 0.4).normalize(),
            ));
        }
        rows
    };
    let first_new = part.pos.len() as u32;
    let edge_uv: Vec<[f32; 2]> = prev.iter().map(|&v| part.uv[v as usize]).collect();
    for row in rows {
        let cur: Vec<u32> = row
            .iter()
            .enumerate()
            .map(|(q, &(p, n))| part.vert(p, n, edge_uv[q], [1.0; 3]))
            .collect();
        for q in 0..cur.len() - 1 {
            part.quad(prev[q], prev[q + 1], cur[q + 1], cur[q], m);
        }
        prev = cur;
    }
    // The hem's vertices join the grid's UV chart (at its edge's UVs).
    if let Some(c) = part.charts.last_mut() {
        if c.first + c.count == first_new {
            c.count = part.pos.len() as u32 - c.first;
        }
    }
}

/// A panel: the strip's grid, its shut-line edges stepped back and hemmed,
/// emitted on both sides when `mirror` (the strip is the left one).
#[allow(clippy::too_many_arguments)]
pub fn panel(
    model: &mut Model,
    name: &str,
    mirror_name: Option<&str>,
    st: &Strip,
    m: Mat,
    gaps: [bool; 4],
    hems: [bool; 4],
) -> Grid {
    panel_by(
        model,
        name,
        mirror_name,
        st,
        m,
        gaps.map(|e| if e { GAP } else { 0.0 }),
        hems,
    )
}

/// [`panel`] with each edge's step-back distance.
pub fn panel_by(
    model: &mut Model,
    name: &str,
    mirror_name: Option<&str>,
    st: &Strip,
    m: Mat,
    gaps: [f32; 4],
    hems: [bool; 4],
) -> Grid {
    let mut g = grid(st, st.h);
    g.shrink_by(gaps);
    for (nm, gg) in [(Some(name), g.clone()), (mirror_name, g.mirrored())] {
        let Some(nm) = nm else { continue };
        let part = model.part(nm);
        let ids = emit(part, &gg, m, [1.0; 3]);
        for e in 0..4 {
            if hems[e] {
                hem(part, &gg, &ids, e, m);
            }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Boundary helpers.
// ---------------------------------------------------------------------------

/// θ of the point `d` metres along the section from θ (positive: toward
/// larger θ).
fn theta_walk(u: f32, th: f32, d: f32) -> f32 {
    let step = 0.002f32.copysign(d);
    let mut t = th;
    let mut p = sf::hit(Param::Section, u, t);
    let mut acc = 0.0;
    for _ in 0..2000 {
        let t2 = t + step;
        let q = sf::hit(Param::Section, u, t2);
        acc += q.distance(p);
        t = t2;
        p = q;
        if acc >= d.abs() {
            break;
        }
    }
    t
}

/// θ on the bottom (left) where the underside reaches half width `x`.
fn theta_at_x_under(u: f32, x: f32) -> f32 {
    let (mut lo, mut hi) = (1.7f32, 3.13f32);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if sf::hit(Param::Section, u, m).x > x {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

fn tab(f: impl Fn(f32) -> f32 + 'static, lo: f32, hi: f32) -> F {
    let t = Table::new(f, lo, hi);
    Box::new(move |x| t.at(x))
}

/// y on a nose or tail ray family's boundary from a list of surface points
/// (sorted by φ): (φ, y) pairs, interpolated.
fn phi_y(param: Param, pts: &[Vec3]) -> F {
    let mut v: Vec<(f32, f32)> = pts.iter().map(|&p| sf::invert(param, p)).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    v.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-5);
    Box::new(move |phi| {
        let k = v.partition_point(|q| q.0 < phi);
        if k == 0 {
            return v[0].1;
        }
        if k >= v.len() {
            return v[v.len() - 1].1;
        }
        let (a, b) = (v[k - 1], v[k]);
        a.1 + (b.1 - a.1) * (phi - a.0) / (b.0 - a.0).max(1e-9)
    })
}

/// The plan angle (nose or tail family) of the surface point at half width
/// `x` and height `y` in front (or behind) the axis.
fn phi_at_x(param: Param, x: f32, y: f32) -> f32 {
    let (mut lo, mut hi) = (0.0f32, FRAC_PI_2);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if sf::hit(param, m, y).x < x {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// Mesh primitives for the hardware.
// ---------------------------------------------------------------------------

/// A rounded box (centre, half extents, edge radius, orientation), `n`
/// cells per face side.
fn rounded_box(part: &mut Part, c: Vec3, half: Vec3, r: f32, rot: Mat3, m: Mat, n: usize) {
    let r = r.min(half.min_element() * 0.999);
    let core = half - Vec3::splat(r);
    for axis in 0..3 {
        for sgn in [-1.0f32, 1.0] {
            let mut ids = vec![0u32; (n + 1) * (n + 1)];
            for j in 0..=n {
                for i in 0..=n {
                    let (u, v) = (
                        -1.0 + 2.0 * i as f32 / n as f32,
                        -1.0 + 2.0 * j as f32 / n as f32,
                    );
                    // Even angular spacing round the rounded edges.
                    let warp = |x: f32| (x * 0.785).tan() / 0.785f32.tan();
                    let mut q = [0.0f32; 3];
                    q[axis] = sgn;
                    q[(axis + 1) % 3] = warp(u);
                    q[(axis + 2) % 3] = warp(v);
                    let q = Vec3::from_array(q) * half;
                    let inner = q.clamp(-core, core);
                    let dn = (q - inner).normalize_or(Vec3::Y);
                    let p = inner + dn * r;
                    ids[j * (n + 1) + i] =
                        part.vert(c + rot * p, rot * dn, [u * half.x, v * half.y], [1.0; 3]);
                }
            }
            for j in 0..n {
                for i in 0..n {
                    let a = ids[j * (n + 1) + i];
                    let b = ids[j * (n + 1) + i + 1];
                    let cc = ids[(j + 1) * (n + 1) + i + 1];
                    let d = ids[(j + 1) * (n + 1) + i];
                    part.quad(a, b, cc, d, m);
                }
            }
        }
    }
}

/// A surface of revolution: `profile` (radius, axial position) about the
/// axis through `c` along `ax` (closed round `seg` segments).
fn revolve(
    part: &mut Part,
    c: Vec3,
    ax: Vec3,
    profile: &[(f32, f32)],
    seg: usize,
    m: Mat,
    _smooth_profile: bool,
) {
    revolve_v(part, c, ax, profile, None, seg, m);
}

/// [`revolve`] with each profile point's v coordinate (u runs 0 … 1 round
/// the axis, its seam duplicated so a texture wraps cleanly).
fn revolve_v(
    part: &mut Part,
    c: Vec3,
    ax: Vec3,
    profile: &[(f32, f32)],
    v: Option<&[f32]>,
    seg: usize,
    m: Mat,
) {
    let ax = ax.normalize();
    let e1 = if ax.y.abs() < 0.9 {
        ax.cross(Vec3::Y).normalize()
    } else {
        ax.cross(Vec3::X).normalize()
    };
    let e2 = ax.cross(e1);
    let np = profile.len();
    // Profile normals (in the (r, w) plane): perpendicular to the tangent.
    let pn: Vec<(f32, f32)> = (0..np)
        .map(|k| {
            let a = profile[k.saturating_sub(1)];
            let b = profile[(k + 1).min(np - 1)];
            let (dr, dw) = (b.0 - a.0, b.1 - a.1);
            let l = (dr * dr + dw * dw).sqrt().max(1e-9);
            (dw / l, -dr / l)
        })
        .collect();
    let mut rings: Vec<Vec<u32>> = Vec::new();
    for k in 0..np {
        let (r, w) = profile[k];
        let (nr, nw) = pn[k];
        let vk = v.map_or(k as f32 / np as f32, |v| v[k]);
        let ring: Vec<u32> = (0..=seg)
            .map(|s| {
                let a = TAU * s as f32 / seg as f32;
                let rad = e1 * a.cos() + e2 * a.sin();
                let n = (rad * nr + ax * nw).normalize_or(rad);
                part.vert(
                    c + rad * r + ax * w,
                    n,
                    [s as f32 / seg as f32, vk],
                    [1.0; 3],
                )
            })
            .collect();
        rings.push(ring);
    }
    for k in 0..np - 1 {
        for s in 0..seg {
            part.quad(
                rings[k][s],
                rings[k][s + 1],
                rings[k + 1][s + 1],
                rings[k + 1][s],
                m,
            );
        }
    }
}
/// A box swept between two points (a spoke, a wiper): width along `side`,
/// thickness along `up`.
fn bar(part: &mut Part, a: Vec3, b: Vec3, up: Vec3, w0: f32, w1: f32, th: f32, m: Mat) {
    let dir = (b - a).normalize();
    let side = dir.cross(up).normalize();
    let up = side.cross(dir);
    let n = 6;
    let ring = |p: Vec3, w: f32| -> [Vec3; 4] {
        [
            p + side * w + up * th,
            p - side * w + up * th,
            p - side * w - up * th,
            p + side * w - up * th,
        ]
    };
    let norms = [up, -side, -up, side];
    let mut prev: Option<[u32; 8]> = None;
    for k in 0..=n {
        let t = k as f32 / n as f32;
        let p = a.lerp(b, t);
        let r = ring(p, w0 + (w1 - w0) * t);
        // Each face its own vertices (hard edges).
        let mut ids = [0u32; 8];
        for f in 0..4 {
            ids[2 * f] = part.vert(r[f], norms[f], [t, 0.0], [1.0; 3]);
            ids[2 * f + 1] = part.vert(r[(f + 1) % 4], norms[f], [t, 1.0], [1.0; 3]);
        }
        if let Some(pv) = prev {
            for f in 0..4 {
                part.quad(pv[2 * f], pv[2 * f + 1], ids[2 * f + 1], ids[2 * f], m);
            }
        }
        prev = Some(ids);
    }
    // End caps.
    for (p, d, w) in [(a, -dir, w0), (b, dir, w1)] {
        let r = ring(p, w);
        let ids: Vec<u32> = r
            .iter()
            .map(|&q| part.vert(q, d, [0.0, 0.0], [1.0; 3]))
            .collect();
        part.quad(ids[0], ids[1], ids[2], ids[3], m);
    }
}

// ---------------------------------------------------------------------------
// The car.
// ---------------------------------------------------------------------------

/// Door shut lines (z) and the door bottom (rocker line) height.
const DOOR_F: f32 = WS_BASE;
const DOOR_B: f32 = -0.345;
const DOOR_R: f32 = -0.96;
const Y_DOOR_BOTTOM: f32 = TYRE_R;
/// Windshield base and top, backlight top and base (z).
const WS_BASE: f32 = 0.745;
const WS_TOP: f32 = -0.08;
const BL_TOP: f32 = -1.00;
const BL_BASE: f32 = -2.05;
/// The hood's rear edge (the cowl between it and the windshield).
const HOOD_REAR: f32 = 0.86;
/// Bumper top on the sides (front, rear).
const BUMPER_TOP_F: f32 = 0.60;
const BUMPER_TOP_R: f32 = 0.78;
/// The lamps' depth below their top edge (slim LED units), in height.
const HEADLAMP_H: f32 = 0.105;
const TAILLAMP_H: f32 = 0.075;

pub fn build() -> Model {
    let mut m = Model::default();
    let belt = |u: f32| sf::with_profiles(|p| p.belt.at(u));
    // ------------------------------------------------------------------
    // Lines in the section family (left side).
    // ------------------------------------------------------------------
    let th_belt = Table::new(move |u| sf::theta_at_y(u, belt(u)), -2.5, 2.4);
    let th_door_bottom = Table::new(|u| sf::theta_at_y(u, Y_DOOR_BOTTOM), -2.5, 2.4);
    // The shoulder's character line (an edge loop on the side panels).
    let th_shoulder = Table::new(|u| sf::theta_at_y_side(u, sf::shoulder_y(u)), -2.5, 2.4);
    // The lower character line (the doors' scallop): an edge loop on the doors.
    let th_scallop = Table::new(|u| sf::theta_at_y(u, sf::scallop_y(u)), -1.2, 1.0);
    // Bottom of the side: the arch edge, else the rocker line.
    let th_side_bottom = Table::new(
        |u| {
            sf::theta_at_y(
                u,
                sf::arch_y(u).map_or(Y_DOOR_BOTTOM, |y| y.max(Y_DOOR_BOTTOM)),
            )
        },
        -2.5,
        2.4,
    );
    // The greenhouse's corner and the lines either side of it: the glass's
    // edge on the top (windshield, backlight, roof edge) and the door frame's
    // top on the side.
    // (Kept above the belt: at the A-pillar's foot the greenhouse fades into
    // the cowl and the 50° lean would land on the body side, below it.)
    let th_corner = {
        let b = th_belt.clone();
        Table::new(
            move |u| sf::theta_corner(u).min(b.at(u) - 0.07),
            -2.15,
            0.85,
        )
    };
    let th_top_edge = {
        let c = th_corner.clone();
        Table::new(move |u| theta_walk(u, c.at(u), -0.05), -2.15, 0.85)
    };
    let th_frame_top = {
        let c = th_corner.clone();
        let b = th_belt.clone();
        Table::new(
            move |u| theta_walk(u, c.at(u), 0.035).min(b.at(u) - 0.02),
            -2.15,
            0.85,
        )
    };
    // The hood and trunk lines on the top.
    let x_hood = |u: f32| 0.80 - 0.26 * sf::smoothstep(1.45, 2.25, u).powf(1.4);
    let th_hood = Table::new(move |u| sf::theta_at_x(u, x_hood(u)), 0.5, 2.4);
    // The backlight sits 15 cm inside the greenhouse's corner (a sedan's
    // broad C-pillars); the trunk lid starts at its base's width.
    // Its edge by half width: 15 cm in from the corner at the top, closing
    // to the trunk lid's width at its base (the corner itself fades into the
    // deck there).
    let x_bl_top = sf::hit(
        Param::Section,
        BL_TOP,
        theta_walk(BL_TOP, th_corner.at(BL_TOP), -0.15),
    )
    .x;
    let x_blb = 0.62;
    let th_bl_edge = Table::new(
        move |u| {
            sf::theta_at_x(
                u,
                x_bl_top + (x_blb - x_bl_top) * sf::smoothstep(BL_TOP, BL_BASE, u),
            )
        },
        -2.15,
        -0.9,
    );
    let x_trunk = move |u: f32| {
        let w = x_blb + (0.76 - x_blb) * sf::smoothstep(BL_BASE, BL_BASE - 0.20, u);
        w - 0.20 * sf::smoothstep(-2.10, -2.48, u).powf(1.4)
    };
    // Clamped inside the deck's own edge (where it turns 45° to the side), which
    // narrows as the tail rounds in plan.
    let x_deck = |u: f32| {
        let (mut lo, mut hi) = (0.05f32, 1.5f32);
        for _ in 0..30 {
            let m = 0.5 * (lo + hi);
            if sf::normal(sf::hit(Param::Section, u, m)).x < 0.7 {
                lo = m;
            } else {
                hi = m;
            }
        }
        sf::hit(Param::Section, u, 0.5 * (lo + hi)).x
    };
    let th_trunk = Table::new(
        move |u| sf::theta_at_x(u, x_trunk(u).min(x_deck(u) - 0.06)),
        -2.5,
        -1.6,
    );

    // ------------------------------------------------------------------
    // Hood: from the cowl to its leading edge (bowed back to the corners).
    // ------------------------------------------------------------------
    let hood_front = |t: f32| 2.225 + 0.07 * (1.0 - (2.0 * t - 1.0).powi(2));
    let hood = {
        let h1 = th_hood.clone();
        let h2 = th_hood.clone();
        Strip::new(
            Param::Section,
            k(HOOD_REAR),
            Box::new(hood_front),
            Box::new(move |u| -h1.at(u)),
            Box::new(move |u| h2.at(u)),
        )
    };
    let g_hood = panel(
        &mut m,
        "hood",
        None,
        &hood,
        Mat::Paint,
        [true; 4],
        [true; 4],
    );
    // Inner panel 35 mm behind the skin.
    {
        let inner = g_hood.offset(-0.035).flipped();
        let part = m.part("hood");
        emit(part, &inset(&inner, 1), Mat::Paint, [1.0; 3]);
    }
    // Cowl (black) between the hood and the windshield.
    {
        let (a, b) = (th_hood.clone(), th_hood.clone());
        let st = Strip::new(
            Param::Section,
            k(WS_BASE),
            k(HOOD_REAR),
            Box::new(move |u| -a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        panel(
            &mut m,
            "body_shell",
            None,
            &st,
            Mat::TrimBlack,
            [false, false, false, true],
            [false; 4],
        );
    }

    // ------------------------------------------------------------------
    // Front fenders: door shut line to the front (over the headlamp), the
    // hood line to the arch.
    // ------------------------------------------------------------------
    // t at the bumper top on the seam section (u = ZC_F).
    let t_seam = {
        let lo = th_hood.at(ZC_F);
        let hi = th_side_bottom.at(ZC_F);
        (sf::theta_at_y(ZC_F, BUMPER_TOP_F + 0.03) - lo) / (hi - lo)
    };
    let fender_front = move |t: f32| {
        if t >= t_seam {
            ZC_F
        } else {
            let w = 1.0 - t / t_seam;
            ZC_F + (2.225 - ZC_F) * w.powf(1.6)
        }
    };
    let fender = {
        let (a, b) = (th_hood.clone(), th_side_bottom.clone());
        let mut s = Strip::new(
            Param::Section,
            k(DOOR_F),
            Box::new(fender_front),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        let c = th_shoulder.clone();
        s.creases.push(Box::new(move |u| c.at(u)));
        s
    };
    let g_fender = panel(
        &mut m,
        "fender_FL",
        Some("fender_FR"),
        &fender,
        Mat::Paint,
        [true; 4],
        [true; 4],
    );

    // ------------------------------------------------------------------
    // Doors: below the belt (skins), the window frames above it.
    // ------------------------------------------------------------------
    let door_skin = |m: &mut Model, z0: f32, z1: f32, l: &str, r: &str| -> Grid {
        let (a, b) = (th_belt.clone(), th_door_bottom.clone());
        let mut st = Strip::new(
            Param::Section,
            k(z0),
            k(z1),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        let c = th_shoulder.clone();
        st.creases.push(Box::new(move |u| c.at(u)));
        let c = th_scallop.clone();
        st.creases.push(Box::new(move |u| c.at(u)));
        let g = panel(m, l, Some(r), &st, Mat::Paint, [true; 4], [true; 4]);
        let inner = inset(&g.offset(-0.045).flipped(), 1);
        emit(m.part(l), &inner, Mat::Paint, [1.0; 3]);
        emit(m.part(r), &inner.mirrored(), Mat::Paint, [1.0; 3]);
        g
    };
    let g_door_f = door_skin(&mut m, DOOR_F, DOOR_B, "door_FL", "door_FR");
    let g_door_r = door_skin(&mut m, DOOR_B, DOOR_R, "door_RL", "door_RR");

    // Window frames and glass, left (then mirrored). The daylight opening's
    // ends: the front door's at the A-pillar, the rear's at the C-pillar
    // (slanted: the rear door frame carries the quarter light).
    let frame_w = 0.018;
    let dlo_rear_belt = -1.42;
    let dlo_rear_top = -0.99;
    // Front door: frame from the mirror sail to the B-pillar.
    let sail_end = 0.58;
    // A window-frame strip on the section family: `gaps` each edge's step
    // back (a 4 mm shut line is GAP a side, or 2 GAP on one side when the
    // other side's panel runs on), hemmed where it steps back.
    let upper = |m: &mut Model,
                 name: &str,
                 mirror: &str,
                 a0: F,
                 a1: F,
                 lo: &Table,
                 hi: &Table,
                 mat: Mat,
                 gaps: [f32; 4],
                 off: f32|
     -> Grid {
        let (l, h) = (lo.clone(), hi.clone());
        let mut st = Strip::new(
            Param::Section,
            a0,
            a1,
            Box::new(move |u| l.at(u)),
            Box::new(move |u| h.at(u)),
        );
        st.off = off;
        if mat == Mat::Glass {
            st.h = 0.03;
        }
        panel_by(m, name, Some(mirror), &st, mat, gaps, gaps.map(|g| g > 0.0))
    };
    // Lines inside the frame: glass top (below the frame top) and the belt
    // molding's top.
    let th_glass_top = {
        let f = th_frame_top.clone();
        Table::new(move |u| theta_walk(u, f.at(u), frame_w), -2.15, 0.85)
    };
    let th_glass_bottom = {
        let b = th_belt.clone();
        Table::new(move |u| theta_walk(u, b.at(u), -0.016), -2.15, 0.85)
    };
    let (g1, g2, z0) = (GAP, 2.0 * GAP, 0.0);
    // Front door upper: sail | glass | rear frame, and the top frame; the
    // belt molding along the glass's foot.
    upper(
        &mut m,
        "door_FL",
        "door_FR",
        k(DOOR_F),
        k(sail_end),
        &th_frame_top,
        &th_belt,
        Mat::TrimBlack,
        [g1, g1, g1, z0],
        0.0,
    );
    upper(
        &mut m,
        "door_FL",
        "door_FR",
        k(sail_end),
        k(DOOR_B + 0.03),
        &th_frame_top,
        &th_glass_top,
        Mat::TrimBlack,
        [g1, z0, z0, z0],
        0.0,
    );
    upper(
        &mut m,
        "door_FL",
        "door_FR",
        k(sail_end),
        k(DOOR_B + 0.03),
        &th_glass_bottom,
        &th_belt,
        Mat::Chrome,
        [z0, g1, z0, z0],
        0.0,
    );
    upper(
        &mut m,
        "door_FL",
        "door_FR",
        k(DOOR_B + 0.03),
        k(DOOR_B),
        &th_frame_top,
        &th_belt,
        Mat::TrimBlack,
        [g1, g1, z0, g1],
        0.0,
    );
    upper(
        &mut m,
        "glass_door_FL",
        "glass_door_FR",
        k(sail_end - 0.002),
        k(DOOR_B + 0.032),
        &th_glass_top,
        &th_glass_bottom,
        Mat::Glass,
        [z0; 4],
        -0.006,
    );
    // Rear door upper: front frame | glass | divider | quarter light | rear
    // frame, slanted along the C-pillar: u = slant(τ) at the fraction τ
    // from the frame's top to the belt.
    let slant_u =
        move |tau: f32, off: f32| dlo_rear_top + (dlo_rear_belt - dlo_rear_top) * tau + off;
    let slant = move |off: f32| -> F { Box::new(move |t: f32| slant_u(t, off)) };
    // The slant for a strip between other lines (`lo`, `hi`): u solves
    // u = slant(τ) with τ the point's fraction from the frame top to the belt.
    let (sft, sbt) = (th_frame_top.clone(), th_belt.clone());
    let slant_in = move |lo: &Table, hi: &Table, off: f32| -> F {
        let (lo, hi, ft, bt) = (lo.clone(), hi.clone(), sft.clone(), sbt.clone());
        Box::new(move |t: f32| {
            let mut u = slant_u(0.5, off);
            for _ in 0..12 {
                let th = lo.at(u) + t * (hi.at(u) - lo.at(u));
                let tau = ((th - ft.at(u)) / (bt.at(u) - ft.at(u))).clamp(0.0, 1.0);
                u = slant_u(tau, off);
            }
            u
        })
    };
    let div_a = -0.95;
    let div_b = -0.975;
    upper(
        &mut m,
        "door_RL",
        "door_RR",
        k(DOOR_B),
        k(DOOR_B - 0.03),
        &th_frame_top,
        &th_belt,
        Mat::TrimBlack,
        [g1, g1, g1, z0],
        0.0,
    );
    upper(
        &mut m,
        "door_RL",
        "door_RR",
        k(DOOR_B - 0.03),
        slant_in(&th_frame_top, &th_glass_top, 0.0),
        &th_frame_top,
        &th_glass_top,
        Mat::TrimBlack,
        [g1, z0, z0, z0],
        0.0,
    );
    // The belt molding runs on over the quarter panel, which has no step
    // back there: the molding takes the whole 4 mm.
    upper(
        &mut m,
        "door_RL",
        "door_RR",
        k(DOOR_B - 0.03),
        slant_in(&th_glass_bottom, &th_belt, 0.0),
        &th_glass_bottom,
        &th_belt,
        Mat::Chrome,
        [z0, g2, z0, z0],
        0.0,
    );
    upper(
        &mut m,
        "door_RL",
        "door_RR",
        k(div_a),
        k(div_b),
        &th_glass_top,
        &th_glass_bottom,
        Mat::TrimBlack,
        [z0; 4],
        0.0,
    );
    upper(
        &mut m,
        "door_RL",
        "door_RR",
        slant(0.0),
        slant(-0.02),
        &th_frame_top,
        &th_belt,
        Mat::TrimBlack,
        [g1, g2, z0, g1],
        0.0,
    );
    upper(
        &mut m,
        "glass_door_RL",
        "glass_door_RR",
        k(DOOR_B - 0.028),
        k(div_a - 0.002),
        &th_glass_top,
        &th_glass_bottom,
        Mat::Glass,
        [z0; 4],
        -0.006,
    );
    upper(
        &mut m,
        "glass_quarter_L",
        "glass_quarter_R",
        k(div_b + 0.002),
        slant_in(&th_glass_top, &th_glass_bottom, 0.002),
        &th_glass_top,
        &th_glass_bottom,
        Mat::Glass,
        [z0; 4],
        -0.006,
    );

    // ------------------------------------------------------------------
    // Body shell: A-pillars, roof, C-pillars, quarters, rockers, cowl sides.
    // ------------------------------------------------------------------
    // Windshield and backlight.
    {
        let (a, b) = (th_top_edge.clone(), th_top_edge.clone());
        let mut st = Strip::new(
            Param::Section,
            k(WS_BASE),
            k(WS_TOP),
            Box::new(move |u| -a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        st.off = -0.002;
        st.h = 0.03;
        panel(
            &mut m,
            "glass_windshield",
            None,
            &st,
            Mat::Glass,
            [false; 4],
            [false; 4],
        );
        let (a, b) = (th_bl_edge.clone(), th_bl_edge.clone());
        let mut st = Strip::new(
            Param::Section,
            k(BL_TOP),
            k(BL_BASE),
            Box::new(move |u| -a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        st.off = -0.002;
        st.h = 0.03;
        panel(
            &mut m,
            "glass_rear",
            None,
            &st,
            Mat::Glass,
            [false; 4],
            [false; 4],
        );
    }
    // A-pillars: windshield edge to the door frame.
    {
        let (a, f) = (th_top_edge.clone(), th_frame_top.clone());
        let st = Strip::new(
            Param::Section,
            k(WS_BASE),
            k(WS_TOP),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| f.at(u)),
        );
        panel(
            &mut m,
            "body_shell",
            Some("body_shell"),
            &st,
            Mat::Paint,
            [false, true, true, false],
            [false, true, true, false],
        );
    }
    // Roof: windshield top to backlight top, frame top to frame top.
    {
        let (a, b) = (th_frame_top.clone(), th_frame_top.clone());
        let st = Strip::new(
            Param::Section,
            k(WS_TOP),
            k(BL_TOP),
            Box::new(move |u| -a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        panel(
            &mut m,
            "body_shell",
            None,
            &st,
            Mat::Paint,
            [true, true, false, false],
            [true, true, false, false],
        );
    }
    // C-pillar and the deck's sides: from the backlight's edge (then the
    // trunk lid's side gap) down to the belt, from the roof's rear edge and
    // the rear door frame's slant to the tail. Its start meets the frame's
    // rear edge exactly: u solves u = slant(τ(u, t)).
    let c_top = {
        let (e, tr) = (th_bl_edge.clone(), th_trunk.clone());
        Table::new(
            move |u| if u > BL_BASE { e.at(u) } else { tr.at(u) },
            -2.5,
            -0.9,
        )
    };
    let quarter_end = -2.36;
    let deck_end: Vec<Vec3>;
    {
        let (a, b) = (c_top.clone(), th_belt.clone());
        let (ft, ct, bt) = (th_frame_top.clone(), c_top.clone(), th_belt.clone());
        let a0: F = Box::new(move |t: f32| {
            let mut u = slant_u(0.5, -0.02);
            for _ in 0..12 {
                let th = ct.at(u) + t * (bt.at(u) - ct.at(u));
                let tau = ((th - ft.at(u)) / (bt.at(u) - ft.at(u))).clamp(0.0, 1.0);
                u = slant_u(tau, -0.02);
            }
            u
        });
        let st = Strip::new(
            Param::Section,
            a0,
            k(BL_BASE),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        panel_by(
            &mut m,
            "body_shell",
            Some("body_shell"),
            &st,
            Mat::Paint,
            [g1, z0, g1, z0],
            [true, false, true, false],
        );
        // The deck's sides, beside the trunk lid, to the tail.
        let (a, b) = (c_top.clone(), th_belt.clone());
        let st = Strip::new(
            Param::Section,
            k(BL_BASE),
            k(quarter_end),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        deck_end = panel_by(
            &mut m,
            "body_shell",
            Some("body_shell"),
            &st,
            Mat::Paint,
            [g1, z0, z0, g1],
            [true, false, false, true],
        )
        .edge_points(3);
    }
    // Quarter below the belt, in two strips: ahead of ZC_R down to the rear
    // arch (the rear door to the bumper's end), and behind it down to the
    // bumper's top, back to the deck side's end (the tail lamp's shut line
    // runs straight down the deck side's and this strip's ends).
    let th_bumper_r = Table::new(|u| sf::theta_at_y(u, BUMPER_TOP_R), -2.45, -1.6);
    {
        let (a, b) = (th_belt.clone(), th_side_bottom.clone());
        let mut s = Strip::new(
            Param::Section,
            k(DOOR_R),
            k(ZC_R),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        let c = th_shoulder.clone();
        s.creases.push(Box::new(move |u| c.at(u)));
        panel_by(
            &mut m,
            "body_shell",
            Some("body_shell"),
            &s,
            Mat::Paint,
            [0.0, GAP, GAP, GAP],
            [false, true, true, true],
        );
    }
    let g_quarter = {
        let (a, b) = (th_belt.clone(), th_bumper_r.clone());
        let mut s = Strip::new(
            Param::Section,
            k(ZC_R),
            k(quarter_end),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        let c = th_shoulder.clone();
        s.creases.push(Box::new(move |u| c.at(u)));
        panel_by(
            &mut m,
            "body_shell",
            Some("body_shell"),
            &s,
            Mat::Paint,
            [0.0, GAP, 0.0, GAP],
            [false, true, false, true],
        )
    };
    // Rockers: between the arches, from the door bottoms under to the floor.
    {
        let a = th_door_bottom.clone();
        let st = Strip::new(
            Param::Section,
            k(AXLE_F - ARCH_R),
            k(AXLE_R + ARCH_R),
            Box::new(move |u| a.at(u)),
            tab(
                |u| theta_at_x_under(u, 0.70),
                AXLE_R + ARCH_R,
                AXLE_F - ARCH_R,
            ),
        );
        panel(
            &mut m,
            "body_shell",
            Some("body_shell"),
            &st,
            Mat::Paint,
            [true, false, true, true],
            [true, false, true, true],
        );
    }

    // ------------------------------------------------------------------
    // Trunk lid: the deck, then its rear face between the tail lamps.
    // ------------------------------------------------------------------
    let deck_rear = |t: f32| -2.425 - 0.04 * (1.0 - (2.0 * t - 1.0).powi(2));
    let trunk = {
        let (a, b) = (th_trunk.clone(), th_trunk.clone());
        Strip::new(
            Param::Section,
            k(BL_BASE),
            Box::new(deck_rear),
            Box::new(move |u| -a.at(u)),
            Box::new(move |u| b.at(u)),
        )
    };
    let mut g_trunk = grid(&trunk, H);
    g_trunk.shrink([true, true, true, false], GAP);
    {
        let part = m.part("trunk_lid");
        let ids = emit(part, &g_trunk, Mat::Paint, [1.0; 3]);
        for e in 0..3 {
            hem(part, &g_trunk, &ids, e, Mat::Paint);
        }
        let inner = inset(&g_trunk.offset(-0.035).flipped(), 1);
        emit(part, &inner, Mat::Paint, [1.0; 3]);
    }
    // The rear face in the tail family: a painted face under the ducktail
    // lip, a slim red light bar across it just under the lip (in line with
    // the tail lamps), the shut line to the bumper below.
    let fold = g_trunk.edge_points(3);
    let fold_t = {
        let f = phi_y(Param::Tail, &fold);
        Table::new(f, -FRAC_PI_2, FRAC_PI_2)
    };
    let x_tl_in = 0.47;
    let phi_tl_in = phi_at_x(Param::Tail, x_tl_in, 0.95);
    let y_rbt = |_phi: f32| BUMPER_TOP_R;
    // The light bar's band (below the fold).
    let (bar0, bar1) = (0.082f32, 0.050f32);
    {
        let (f0, f1, f2, f3) = (
            fold_t.clone(),
            fold_t.clone(),
            fold_t.clone(),
            fold_t.clone(),
        );
        let pieces: [(F, F, [bool; 4]); 2] = [
            (
                Box::new(y_rbt),
                Box::new(move |phi| f0.at(phi) - bar0),
                [true, true, true, true],
            ),
            (
                Box::new(move |phi| f1.at(phi) - bar1),
                Box::new(move |phi| f2.at(phi)),
                [true, false, true, true],
            ),
        ];
        for (lo, hi, gaps) in pieces {
            let st = Strip::new(Param::Tail, k(-phi_tl_in), k(phi_tl_in), lo, hi);
            let mut g = grid(&st, H);
            g.shrink(gaps, GAP);
            let part = m.part("trunk_lid");
            let ids = emit(part, &g, Mat::Paint, [1.0; 3]);
            for e in 0..4 {
                if gaps[e] {
                    hem(part, &g, &ids, e, Mat::Paint);
                }
            }
        }
        let st = Strip::new(
            Param::Tail,
            k(-phi_tl_in),
            k(phi_tl_in),
            Box::new(move |phi| f3.at(phi) - bar0),
            {
                let f = fold_t.clone();
                Box::new(move |phi| f.at(phi) - bar1)
            },
        );
        let mut g = grid(&st, 0.012);
        g.shrink([true; 4], GAP);
        lamp(
            &mut m,
            "trunk_lid",
            "trunk_lid",
            &g,
            [0.75, 0.03, 0.03],
            LampKind::Bar,
        );
    }
    // Tail lamps: slim units at the rear face's corners, wrapping round onto
    // the side as far as the quarter's straight end (z = quarter_end), their
    // tops tucked under one continuous shut line: the boot lid's fold, then
    // the deck side's end. They taper at their inner ends to the light bar's
    // height. Under them a body panel runs down to the bumper.
    let phi_q = {
        // The quarter's end in the tail family: its points' mean plan angle.
        let e = g_quarter.edge_points(3);
        e.iter().map(|&p| sf::invert(Param::Tail, p).0).sum::<f32>() / e.len() as f32
    };
    let tl_top = {
        let mut top: Vec<Vec3> = fold.iter().copied().filter(|p| p.x > 0.0).collect();
        top.extend(deck_end.iter().copied());
        // The deck side's end below the belt continues as the quarter's.
        top.extend(g_quarter.edge_points(3));
        Table::new(phi_y(Param::Tail, &top), 0.0, FRAC_PI_2)
    };
    let tl_lo = {
        let t = tl_top.clone();
        Table::new(
            move |phi| {
                let h =
                    0.088 + (TAILLAMP_H - 0.088) * sf::smoothstep(phi_tl_in, phi_tl_in + 0.35, phi);
                (t.at(phi) - h).max(y_rbt(phi) + 0.035)
            },
            0.0,
            FRAC_PI_2,
        )
    };
    {
        let (a, b) = (tl_lo.clone(), tl_top.clone());
        let st = Strip::new(
            Param::Tail,
            k(phi_tl_in),
            k(phi_q),
            Box::new(move |p| a.at(p)),
            Box::new(move |p| b.at(p)),
        );
        let mut g = grid(&st, 0.014);
        g.shrink([true, true, true, true], GAP);
        lamp(
            &mut m,
            "taillight_L_lens",
            "taillight_L_housing",
            &g,
            [0.75, 0.03, 0.03],
            LampKind::Tail,
        );
        let g = g.mirrored();
        lamp(
            &mut m,
            "taillight_R_lens",
            "taillight_R_housing",
            &g,
            [0.75, 0.03, 0.03],
            LampKind::Tail,
        );
        // The rear face under the lamp, to the bumper.
        let a = tl_lo.clone();
        let st = Strip::new(
            Param::Tail,
            k(phi_tl_in),
            k(phi_q),
            Box::new(y_rbt),
            Box::new(move |p| a.at(p)),
        );
        panel(
            &mut m,
            "body_shell",
            Some("body_shell"),
            &st,
            Mat::Paint,
            [true, true, true, true],
            [true, true, true, true],
        );
    }
    // ------------------------------------------------------------------
    // Nose: grille, headlamps, bumper.
    // ------------------------------------------------------------------
    let hood_edge = g_hood.edge_points(3);
    let fender_edge = g_fender.edge_points(3);
    let x_grille = 0.40;
    let phi_g = phi_at_x(Param::Nose, x_grille, 0.62);
    // Headlamps: slim units under the bonnet's and the fender's edges.
    let hl_top = {
        let mut top: Vec<Vec3> = hood_edge.iter().copied().filter(|p| p.x > 0.0).collect();
        top.extend(fender_edge.iter().copied());
        Table::new(phi_y(Param::Nose, &top), 0.0, FRAC_PI_2)
    };
    let hl_lo = {
        let t = hl_top.clone();
        Table::new(
            move |phi| (t.at(phi) - HEADLAMP_H).max(BUMPER_TOP_F),
            -FRAC_PI_2,
            FRAC_PI_2,
        )
    };
    let hl_lo_abs = {
        let t = hl_lo.clone();
        move |phi: f32| t.at(phi.abs())
    };
    {
        let (a, b) = (hl_lo.clone(), hl_top.clone());
        let st = Strip::new(
            Param::Nose,
            k(phi_g),
            k(FRAC_PI_2 - 1e-3),
            Box::new(move |p| a.at(p)),
            Box::new(move |p| b.at(p)),
        );
        let mut g = grid(&st, 0.014);
        g.shrink([true, true, true, true], GAP);
        lamp(
            &mut m,
            "headlight_L_lens",
            "headlight_L_housing",
            &g,
            [1.0; 3],
            LampKind::Head,
        );
        lamp(
            &mut m,
            "headlight_R_lens",
            "headlight_R_housing",
            &g.mirrored(),
            [1.0; 3],
            LampKind::Head,
        );
    }
    // Between the lamps: a painted nose bar carrying a slim LED light bar
    // under the bonnet's edge (the lamps' running lights joined across the
    // car). Below it the grille, a trapezoid widening downward (a slim
    // dark-chrome frame round a recessed honeycomb), a painted bar, then a
    // broad lower intake over the splitter. The centre's sides follow one
    // line, `x_edge(y)`.
    let (y_bb, y_it, y_gb, y_gt) = (0.30f32, 0.352f32, 0.40f32, 0.60f32);
    let x_edge = move |y: f32| {
        if y <= y_it {
            0.48
        } else if y <= y_gb {
            0.48 - 0.02 * (y - y_it) / (y_gb - y_it)
        } else {
            0.46 - 0.05 * ((y - y_gb) / (y_gt - y_gb)).min(1.0)
        }
    };
    let phi_e = Table::new(
        move |y| phi_at_x(Param::Nose, x_edge(y), y),
        y_bb - 0.02,
        y_gt + 0.02,
    );
    // A centre strip between heights y0 and y1, its ends on the line.
    let centre = |y0: f32, y1: f32| -> Strip {
        let (pa, pb) = (phi_e.clone(), phi_e.clone());
        Strip::new(
            Param::Nose,
            Box::new(move |t| -pa.at(y0 + t * (y1 - y0))),
            Box::new(move |t| pb.at(y0 + t * (y1 - y0))),
            k(y0),
            k(y1),
        )
    };
    {
        let edge = Table::new(phi_y(Param::Nose, &hood_edge), -phi_g, phi_g);
        let (e0, e1, e2) = (edge.clone(), edge.clone(), edge.clone());
        let st = Strip::new(
            Param::Nose,
            k(-phi_g),
            k(phi_g),
            k(y_gt),
            Box::new(move |p| e0.at(p) - 0.026),
        );
        panel(
            &mut m,
            "bumper_front",
            None,
            &st,
            Mat::Paint,
            [true, true, false, false],
            [true, true, false, false],
        );
        let st = Strip::new(
            Param::Nose,
            k(-phi_g),
            k(phi_g),
            Box::new(move |p| e1.at(p) - 0.026),
            Box::new(move |p| e2.at(p)),
        );
        let mut g = grid(&st, 0.012);
        g.shrink([true; 4], GAP);
        lamp(&mut m, "grille", "grille", &g, [1.0; 3], LampKind::Bar);
        let mut g = grid(&centre(y_gb, y_gt), 0.016);
        g.shrink([true, true, true, true], GAP);
        grille(&mut m, &g);
        // The bar between the grille and the intake.
        panel(
            &mut m,
            "bumper_front",
            None,
            &centre(y_it, y_gb),
            Mat::Paint,
            [true, true, false, false],
            [true, true, false, false],
        );
        // The lower intake: recessed, textured black.
        let g = grid(&centre(y_bb + 0.004, y_it), H);
        recess(
            &mut m,
            "bumper_front",
            &g,
            0.035,
            Mat::TrimBlack,
            Mat::TrimBlack,
        );
    }
    // Bumper: beside the centre, under the lamps and round to the arches; a
    // corner vent each side; a textured black splitter along its foot.
    let phi_v0 = phi_at_x(Param::Nose, 0.665, 0.42);
    let phi_v1 = phi_at_x(Param::Nose, 0.80, 0.42);
    let (yv0, yv1) = (0.335, 0.53);
    {
        let side = |lo: F, hi: F, a0: F, a1: F, gaps: [bool; 4]| (a0, a1, lo, hi, gaps);
        let al = || arch_lo(Param::Nose, y_bb);
        let hl = || -> F { Box::new(hl_lo_abs.clone()) };
        // The inner side follows the centre's line: φ at the height of the
        // point t up the strip (lo and hi themselves depend on φ: solved).
        let inner: F = {
            let (pe, lo, hi) = (phi_e.clone(), arch_lo(Param::Nose, y_bb), hl_lo_abs.clone());
            Box::new(move |t| {
                let mut phi = pe.at(y_gb);
                for _ in 0..8 {
                    let y = lo(phi) + t * (hi(phi) - lo(phi));
                    phi = if y >= y_gt { phi_g } else { pe.at(y) };
                }
                phi
            })
        };
        let pieces: Vec<(F, F, F, F, [bool; 4])> = vec![
            side(al(), hl(), inner, k(phi_v0), [true, true, true, false]),
            side(
                al(),
                k(yv0),
                k(phi_v0),
                k(phi_v1),
                [true, false, false, false],
            ),
            side(
                k(yv1),
                hl(),
                k(phi_v0),
                k(phi_v1),
                [false, true, false, false],
            ),
            side(
                al(),
                hl(),
                k(phi_v1),
                k(FRAC_PI_2),
                [true, true, false, true],
            ),
        ];
        for (a0, a1, lo, hi, gaps) in pieces {
            let st = Strip::new(Param::Nose, a0, a1, lo, hi);
            let mut g = grid(&st, H);
            g.shrink(gaps, GAP);
            let part = m.part("bumper_front");
            for gg in [g.clone(), g.mirrored()] {
                let ids = emit(part, &gg, Mat::Paint, [1.0; 3]);
                for e in 0..4 {
                    if gaps[e] {
                        hem(part, &gg, &ids, e, Mat::Paint);
                    }
                }
            }
        }
        // Corner vents (recessed, textured black).
        let st = Strip::new(Param::Nose, k(phi_v0), k(phi_v1), k(yv0), k(yv1));
        let g = grid(&st, H);
        recess(
            &mut m,
            "bumper_front",
            &g,
            0.03,
            Mat::TrimBlack,
            Mat::TrimBlack,
        );
        recess(
            &mut m,
            "bumper_front",
            &g.mirrored(),
            0.03,
            Mat::TrimBlack,
            Mat::TrimBlack,
        );
        // Splitter: a textured black lip under the bumper's foot, standing
        // proud of it.
        let st = Strip::new(Param::Nose, k(-1.05), k(1.05), k(0.258), k(y_bb - 0.004));
        let g = grid(&st, H).offset(0.012);
        emit(m.part("bumper_front"), &g, Mat::TrimBlack, [1.0; 3]);
    }
    // Rear bumper: the painted cover with the licence plate's recess, over a
    // textured black diffuser with strakes and the exhaust's tips.
    {
        let yd = |phi: f32| 0.47 - 0.09 * sf::smoothstep(0.75, 1.15, phi.abs());
        let blo = Table::new(
            move |phi| {
                let b = yd(phi);
                let b = b + (TYRE_R - b).max(0.0) * sf::smoothstep(1.05, 1.35, phi.abs());
                b.max(arch_at(Param::Tail, phi))
            },
            -FRAC_PI_2,
            FRAC_PI_2,
        );
        let bl = || -> F {
            let t = blo.clone();
            Box::new(move |p| t.at(p))
        };
        // A US plate (305 × 152 mm) on a 330 × 165 mm pocket, 12 mm deep.
        let phi_p = phi_at_x(Param::Tail, 0.165, 0.62);
        let (yp0, yp1) = (0.545, 0.710);
        let pieces: [(f32, f32, F, F, [bool; 4]); 4] = [
            (
                -FRAC_PI_2,
                -phi_p,
                bl(),
                Box::new(y_rbt),
                [true, true, true, false],
            ),
            (
                phi_p,
                FRAC_PI_2,
                bl(),
                Box::new(y_rbt),
                [true, true, false, true],
            ),
            (-phi_p, phi_p, bl(), k(yp0), [true, false, false, false]),
            (
                -phi_p,
                phi_p,
                k(yp1),
                Box::new(y_rbt),
                [false, true, false, false],
            ),
        ];
        for (a0, a1, lo, hi, gaps) in pieces {
            let st = Strip::new(Param::Tail, k(a0), k(a1), lo, hi);
            let mut g = grid(&st, H);
            g.shrink(gaps, GAP);
            let part = m.part("bumper_rear");
            let ids = emit(part, &g, Mat::Paint, [1.0; 3]);
            for e in 0..4 {
                if gaps[e] {
                    hem(part, &g, &ids, e, Mat::Paint);
                }
            }
        }
        let st = Strip::new(Param::Tail, k(-phi_p), k(phi_p), k(yp0), k(yp1));
        plate(&mut m, "bumper_rear", &grid(&st, H), 0.012);
        // Diffuser.
        let st = Strip::new(
            Param::Tail,
            k(-1.15),
            k(1.15),
            k(0.29),
            Box::new(move |phi| yd(phi) - 0.004),
        );
        let g = grid(&st, H).offset(0.002);
        emit(m.part("bumper_rear"), &g, Mat::TrimBlack, [1.0; 3]);
        // The diffuser's surround: a satin-metal strip along the painted
        // cover's foot.
        let st = Strip::new(
            Param::Tail,
            k(-1.02),
            k(1.02),
            Box::new(move |phi| yd(phi) + 0.003),
            Box::new(move |phi| yd(phi) + 0.013),
        );
        let g = grid(&st, H).offset(0.0025);
        emit(m.part("bumper_rear"), &g, Mat::Chrome, [0.55, 0.56, 0.58]);
        // Side reflectors: slim red strips low on the bumper's corners.
        for g in {
            let st = Strip::new(Param::Tail, k(1.08), k(1.42), k(0.505), k(0.523));
            let g = grid(&st, 0.012).offset(0.0025);
            [g.clone(), g.mirrored()]
        } {
            emit(m.part("bumper_rear"), &g, Mat::LampLens, [0.75, 0.03, 0.03]);
        }
        // Strakes: vertical fins standing 24 mm off the diffuser.
        let part = m.part("bumper_rear");
        for x in [-0.30f32, -0.13, 0.13, 0.30] {
            let at = |y: f32| {
                let p = sf::hit(Param::Tail, phi_at_x(Param::Tail, x.abs(), y), y);
                let p = Vec3::new(x.signum() * p.x, p.y, p.z);
                let n = sf::normal(Vec3::new(p.x.abs(), p.y, p.z));
                (p, Vec3::new(x.signum() * n.x, n.y, n.z))
            };
            let (a, na) = at(0.40);
            let (b, nb) = at(0.27);
            bar(
                part,
                a + na * 0.012,
                b + nb * 0.012,
                (na + nb).normalize(),
                0.0045,
                0.0045,
                0.012,
                Mat::TrimBlack,
            );
        }
    }

    // ------------------------------------------------------------------
    // Underbody, wheel wells, wheels, hardware, interior, bay, markers.
    // ------------------------------------------------------------------
    underbody(&mut m);
    wheel_wells(&mut m);
    wheels(&mut m);
    mirrors_handles_wipers(&mut m, &g_door_f, &g_door_r);
    interior(&mut m);
    bay(&mut m);
    markers(&mut m);
    let _ = (g_door_f, g_door_r);
    m
}

/// The wheel arch's edge height on a nose or tail ray φ (0 off the arch):
/// the ray's z taken at the arch's own height (iterated).
fn arch_at(param: Param, phi: f32) -> f32 {
    let mut y = 0.45f32;
    let mut out = 0.0;
    for _ in 0..6 {
        let z = sf::hit(param, phi, y).z;
        match sf::arch_y(z) {
            Some(a) => {
                out = a;
                y = a.clamp(0.2, 0.9);
            }
            None => {
                out = 0.0;
                break;
            }
        }
    }
    out
}

/// A nose or tail panel's lower edge: `base`, rising (to the wheel centre's
/// height at the sides) and following the wheel arch where the panel
/// reaches it.
fn arch_lo(param: Param, base: f32) -> F {
    tab(
        move |phi| {
            let b = base + (TYRE_R - base).max(0.0) * sf::smoothstep(1.05, 1.35, phi.abs());
            b.max(arch_at(param, phi))
        },
        -FRAC_PI_2,
        FRAC_PI_2,
    )
}

fn th_belt_at(u: f32) -> f32 {
    sf::theta_at_y(u, sf::with_profiles(|p| p.belt.at(u)))
}

/// The grid without its outer `c` rings of cells (inner panels).
fn inset(g: &Grid, c: usize) -> Grid {
    let c = c
        .min(g.ns.saturating_sub(1) / 2)
        .min(g.nb.saturating_sub(1) / 2);
    let (ns, nb) = (g.ns - 2 * c, g.nb - 2 * c);
    let pick = |f: &dyn Fn(usize) -> ()| {
        let _ = f;
    };
    let _ = pick;
    let idx: Vec<usize> = (c..=c + nb)
        .flat_map(|j| (c..=c + ns).map(move |i| (i, j)))
        .map(|(i, j)| g.at(i, j))
        .collect();
    Grid {
        ns,
        nb,
        p: idx.iter().map(|&k| g.p[k]).collect(),
        n: idx.iter().map(|&k| g.n[k]).collect(),
        uv: idx.iter().map(|&k| g.uv[k]).collect(),
    }
}

/// What a lamp holds behind its lens.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LampKind {
    /// A headlamp: an L-shaped LED daytime-running-light guide along its top
    /// and down its inner end, two projector modules.
    Head,
    /// A tail lamp: two LED light guides (top and bottom rows).
    Tail,
    /// A light bar: one LED guide along its middle.
    Bar,
}

/// A lamp: its lens on the surface, a black housing behind it (no chrome
/// reflector: LED units), the throat between, LED light guides (emissive)
/// and, in a headlamp, projector modules.
fn lamp(
    m: &mut Model,
    lens_part: &str,
    housing_part: &str,
    g: &Grid,
    tint: [f32; 3],
    kind: LampKind,
) {
    let lens = g.offset(0.0015);
    emit(m.part(lens_part), &lens, Mat::LampLens, tint);
    let depth = 0.04;
    let back = g.offset(-depth).flipped();
    let part = m.part(housing_part);
    // (In the grille, black trim is its honeycomb: the bar's back is the
    // frame's dark chrome.)
    if housing_part.starts_with("grille") {
        emit(part, &back, Mat::Chrome, [0.10, 0.10, 0.11]);
    } else {
        emit(part, &back, Mat::TrimBlack, [1.0; 3]);
    }
    throat(part, g, depth, Mat::TrimBlack);
    let glow = if kind == LampKind::Head {
        [1.0, 0.98, 0.95]
    } else {
        tint
    };
    // A light guide: a ribbon 5 mm wide, 8 mm behind the lens, along a run
    // of grid points, its width toward `toward` (the next row or column).
    let mut guide = |pts: &[(usize, usize)]| {
        let mut prev: Option<(u32, u32)> = None;
        for &(k, toward) in pts {
            let n = g.n[k];
            let w = (g.p[toward] - g.p[k]).normalize_or_zero() * 0.007;
            let p = g.p[k] - n * 0.008;
            let a = part.vert(p, n, [0.0; 2], glow);
            let b = part.vert(p + w, n, [0.0; 2], glow);
            if let Some((pa, pb)) = prev {
                part.quad(pa, a, b, pb, Mat::LampEmissive);
            }
            prev = Some((a, b));
        }
    };
    let (ns, nb) = (g.ns, g.nb);
    let row = |j: usize, dj: isize| -> Vec<(usize, usize)> {
        (1..ns)
            .map(|i| (g.at(i, j), g.at(i, (j as isize + dj) as usize)))
            .collect()
    };
    match kind {
        LampKind::Head => {
            let jt = nb.saturating_sub(1).max(1);
            guide(&row(jt, -1));
            // Down the inner end (s = 0), from the top guide.
            let col: Vec<(usize, usize)> = (1..=jt)
                .rev()
                .map(|j| (g.at(1, j), g.at(2.min(ns), j)))
                .collect();
            guide(&col);
        }
        LampKind::Tail => {
            let jt = nb.saturating_sub(1).max(1);
            guide(&row(jt, -1));
            if nb >= 3 {
                guide(&row(1, 1));
            }
        }
        LampKind::Bar => {
            let j = nb / 2;
            guide(&row(j, if j > 0 { -1 } else { 1 }));
        }
    }
    if kind == LampKind::Head {
        // Projector modules on the middle row: a bright bezel ring round a
        // dark glass lens, 12 mm behind the outer lens.
        let j = nb / 2;
        for f in [0.30f32, 0.50] {
            let i = ((ns as f32 * f).round() as usize).clamp(1, ns - 1);
            let k = g.at(i, j);
            let n = g.n[k];
            let c = g.p[k] - n * 0.012;
            let r = (HEADLAMP_H * 0.30).min(0.022);
            let bezel = [(r, -0.025), (r, 0.0), (r * 0.82, 0.002), (r * 0.70, 0.0)];
            revolve(part, c, n, &bezel, 32, Mat::Chrome, true);
            let lens = [(0.0, 0.004), (r * 0.45, 0.0035), (r * 0.72, 0.001)];
            revolve(part, c, n, &lens, 32, Mat::Glass, true);
        }
    }
}

/// Walls from a grid's border down `depth` along its normals.
fn throat(part: &mut Part, g: &Grid, depth: f32, m: Mat) {
    let mut ring: Vec<usize> = Vec::new();
    ring.extend(g.edge(0));
    ring.extend(g.edge(3).into_iter().skip(1));
    ring.extend(g.edge(1).into_iter().rev().skip(1));
    ring.extend(g.edge(2).into_iter().rev().skip(1));
    let c = g.p.iter().copied().sum::<Vec3>() / g.p.len() as f32;
    let top: Vec<u32> = ring
        .iter()
        .map(|&k| {
            let out = (g.p[k] - c).normalize_or_zero();
            part.vert(g.p[k], -out, [0.0; 2], [1.0; 3])
        })
        .collect();
    let bot: Vec<u32> = ring
        .iter()
        .map(|&k| {
            let out = (g.p[k] - c).normalize_or_zero();
            part.vert(g.p[k] - g.n[k] * depth, -out, [0.0; 2], [1.0; 3])
        })
        .collect();
    for q in 0..ring.len() - 1 {
        part.quad(top[q], top[q + 1], bot[q + 1], bot[q], m);
    }
}

/// An opening: a recessed insert `depth` behind the grid and the throat.
fn recess(m: &mut Model, name: &str, g: &Grid, depth: f32, insert: Mat, wall: Mat) {
    let part = m.part(name);
    emit(part, &g.offset(-depth), insert, [1.0; 3]);
    throat(part, g, depth, wall);
}

/// The grille: a slim dark-chrome frame (the outer ring of cells, tinted
/// near black) round a recessed honeycomb insert.
fn grille(m: &mut Model, g: &Grid) {
    let part = m.part("grille");
    let ring = 1usize.min(g.ns / 2).min(g.nb / 2);
    let dark = [0.10, 0.10, 0.11];
    let vid: Vec<u32> = (0..g.p.len())
        .map(|k| part.vert(g.p[k] + g.n[k] * 0.002, g.n[k], g.uv[k], dark))
        .collect();
    for j in 0..g.nb {
        for i in 0..g.ns {
            if i < ring || j < ring || i >= g.ns - ring || j >= g.nb - ring {
                part.quad(
                    vid[g.at(i, j)],
                    vid[g.at(i + 1, j)],
                    vid[g.at(i + 1, j + 1)],
                    vid[g.at(i, j + 1)],
                    Mat::Chrome,
                );
            }
        }
    }
    let inner = inset(&g.offset(0.002), ring);
    let depth = 0.03;
    emit(part, &inner.offset(-depth), Mat::TrimBlack, [1.0; 3]);
    throat(part, &inner, depth, Mat::TrimBlack);
}

/// A licence plate's pocket: the plate `depth` behind the grid (its UV the
/// plate's own 0 … 1, reading left to right from behind the car) and the
/// pocket's walls.
fn plate(m: &mut Model, name: &str, g: &Grid, depth: f32) {
    let part = m.part(name);
    let mut q = g.offset(-depth);
    for j in 0..=q.nb {
        for i in 0..=q.ns {
            let k = q.at(i, j);
            q.uv[k] = [1.0 - i as f32 / q.ns as f32, j as f32 / q.nb as f32];
        }
    }
    emit(part, &q, Mat::MetalBare, [1.0; 3]);
    // Its own UVs: not packed into the atlas.
    part.charts.pop();
    throat(part, g, depth, Mat::Paint);
}
fn underbody(m: &mut Model) {
    let part = m.part("underbody");
    // A flat pan just above the clearance plane, inside the rockers.
    let (nz, nx) = (180usize, 40usize);
    let mut ids = vec![0u32; (nz + 1) * (nx + 1)];
    for j in 0..=nx {
        for i in 0..=nz {
            let z = 2.10 - 4.35 * i as f32 / nz as f32;
            let x = -0.70 + 1.40 * j as f32 / nx as f32;
            ids[j * (nz + 1) + i] = part.vert(Vec3::new(x, 0.142, z), -Vec3::Y, [z, x], [1.0; 3]);
        }
    }
    for j in 0..nx {
        for i in 0..nz {
            part.quad(
                ids[j * (nz + 1) + i],
                ids[j * (nz + 1) + i + 1],
                ids[(j + 1) * (nz + 1) + i + 1],
                ids[(j + 1) * (nz + 1) + i],
                Mat::TrimBlack,
            );
        }
    }
}

fn wheel_name(front: bool, left: bool) -> &'static str {
    match (front, left) {
        (true, true) => "FL",
        (true, false) => "FR",
        (false, true) => "RL",
        (false, false) => "RR",
    }
}

fn wheel_wells(m: &mut Model) {
    for front in [true, false] {
        for left in [true, false] {
            let s = if left { 1.0 } else { -1.0 };
            let za = if front { AXLE_F } else { AXLE_R };
            let part = m.part(&format!("wheel_well_{}", wheel_name(front, left)));
            let r = ARCH_R + 0.012;
            let (na, nx) = (48usize, 8usize);
            let mut ids = vec![0u32; (na + 1) * (nx + 1)];
            for j in 0..=nx {
                for i in 0..=na {
                    let a = -0.15 + (PI + 0.30) * i as f32 / na as f32;
                    let x = 0.56 + (0.86 - 0.56) * j as f32 / nx as f32;
                    let p = Vec3::new(s * x, TYRE_R + r * a.sin(), za + r * a.cos());
                    let n = -Vec3::new(0.0, a.sin(), a.cos());
                    ids[j * (na + 1) + i] = part.vert(p, n, [a, x], [1.0; 3]);
                }
            }
            for j in 0..nx {
                for i in 0..na {
                    part.quad(
                        ids[j * (na + 1) + i],
                        ids[j * (na + 1) + i + 1],
                        ids[(j + 1) * (na + 1) + i + 1],
                        ids[(j + 1) * (na + 1) + i],
                        Mat::TrimBlack,
                    );
                }
            }
            // Inner wall.
            let c = Vec3::new(s * 0.56, TYRE_R, za);
            let prof: Vec<(f32, f32)> = (0..=6).map(|q| (r * q as f32 / 6.0, 0.0)).collect();
            let _ = prof;
            let ring: Vec<u32> = (0..=na)
                .map(|i| {
                    let a = -0.15 + (PI + 0.30) * i as f32 / na as f32;
                    part.vert(
                        c + Vec3::new(0.0, r * a.sin(), r * a.cos()),
                        Vec3::X * s,
                        [0.0; 2],
                        [1.0; 3],
                    )
                })
                .collect();
            let hub = part.vert(c - Vec3::Y * 0.02, Vec3::X * s, [0.0; 2], [1.0; 3]);
            for i in 0..na {
                part.tri([hub, ring[i], ring[i + 1]], Mat::TrimBlack);
            }
        }
    }
}

fn wheels(m: &mut Model) {
    for front in [true, false] {
        for left in [true, false] {
            let s = if left { 1.0f32 } else { -1.0 };
            let za = if front { AXLE_F } else { AXLE_R };
            let c = Vec3::new(s * TRACK_HALF, TYRE_R, za);
            let ax = Vec3::X * s;
            let tag = wheel_name(front, left);
            let hw = TYRE_W / 2.0;
            // Tyre: tread, rounded shoulders, bulged sidewalls, beads.
            {
                let part = m.part(&format!("tyre_{tag}"));
                let mut prof: Vec<(f32, f32)> = Vec::new();
                let bead = RIM_R + 0.012;
                // Inner bead to the inner shoulder.
                for q in 0..=8 {
                    let t = q as f32 / 8.0;
                    let r = bead + (TYRE_R - 0.03 - bead) * t;
                    let bulge = 0.012 * (PI * t).sin();
                    prof.push((r, -(hw - 0.012) - bulge));
                }
                for q in 1..=6 {
                    let a = PI * (1.0 - q as f32 / 12.0);
                    prof.push((
                        TYRE_R - 0.03 + 0.03 * a.sin(),
                        -hw + 0.03 + 0.03 * a.cos() - 0.0,
                    ));
                }
                for q in 1..=14 {
                    let w = -hw + 0.03 + (2.0 * hw - 0.06) * q as f32 / 14.0;
                    // Shallow tread grooves (circumferential).
                    let groove = if (q % 4) == 2 { 0.004 } else { 0.0 };
                    prof.push((TYRE_R - groove, w));
                }
                for q in 1..=6 {
                    let a = PI * (0.5 - q as f32 / 12.0);
                    prof.push((TYRE_R - 0.03 + 0.03 * a.cos(), hw - 0.03 + 0.03 * a.sin()));
                }
                for q in 1..=8 {
                    let t = 1.0 - q as f32 / 8.0;
                    let r = bead + (TYRE_R - 0.03 - bead) * t;
                    let bulge = 0.012 * (PI * t).sin();
                    prof.push((r, hw - 0.012 + bulge));
                }
                // Profile winds inner bead → outer bead with +w outward.
                // v: the renderer's tyre convention: the outer sidewall's
                // radial fraction (0 at the bead; its lettering band 0.5 …
                // 0.8), the tread 2 … 4 across (shoulder blocks at its
                // edges), the inner sidewall 1 … 1.9 (plain).
                let sw = TYRE_R - 0.03 - bead;
                let v: Vec<f32> = prof
                    .iter()
                    .enumerate()
                    .map(|(q, &(r, w))| {
                        let fr = ((r - bead) / sw).clamp(0.0, 1.0);
                        if q <= 8 {
                            1.0 + 0.9 * fr
                        } else if q <= 34 {
                            2.0 + 2.0 * ((w + hw) / (2.0 * hw)).clamp(0.0, 1.0)
                        } else {
                            fr
                        }
                    })
                    .collect();
                revolve_v(part, c, ax, &prof, Some(&v), 96, Mat::Rubber);
            }
            // Rim: barrel, outer lip, a five-spoke face, hub, lug nuts.
            {
                let part = m.part(&format!("rim_{tag}"));
                let barrel: Vec<(f32, f32)> = vec![
                    (RIM_R - 0.004, -hw + 0.012),
                    (RIM_R + 0.008, -hw + 0.008),
                    (RIM_R + 0.008, -hw + 0.016),
                    (RIM_R - 0.012, -hw + 0.03),
                    (RIM_R - 0.022, 0.0),
                    (RIM_R - 0.012, hw - 0.04),
                    (RIM_R + 0.002, hw - 0.022),
                    (RIM_R + 0.010, hw - 0.014),
                    (RIM_R + 0.010, hw - 0.004),
                    (RIM_R - 0.006, hw - 0.002),
                    (RIM_R - 0.020, hw - 0.012),
                ];
                // The barrel's inside faces the axle: flip the profile normals
                // by winding it from outside in.
                let barrel: Vec<(f32, f32)> = barrel.into_iter().rev().collect();
                revolve(part, c, ax, &barrel, 72, Mat::MetalBare, true);
                // Hub: a concave face (the spokes dish out to the lip), a
                // centre cap.
                let face = hw - 0.05;
                let hub: Vec<(f32, f32)> = vec![
                    (0.0, face + 0.010),
                    (0.028, face + 0.010),
                    (0.034, face + 0.006),
                    (0.062, face + 0.002),
                    (0.074, face - 0.016),
                ];
                revolve(part, c, ax, &hub, 48, Mat::MetalBare, true);
                // Spokes: five split (Y) spokes from the hub to the lip, each
                // arm gunmetal with a diamond-cut face (`chrome` on a rim).
                let e1 = Vec3::Y;
                let e2 = ax.cross(e1);
                for k in 0..5 {
                    let a = TAU * k as f32 / 5.0;
                    for sgn in [-1.0f32, 1.0] {
                        let a0 = a + sgn * 0.07;
                        let a1 = a + sgn * 0.20;
                        let d0 = e1 * a0.cos() + e2 * a0.sin();
                        let d1 = e1 * a1.cos() + e2 * a1.sin();
                        let p0 = c + d0 * 0.062 + ax * (face - 0.004);
                        let p1 = c + d1 * (RIM_R - 0.014) + ax * (hw - 0.020);
                        bar(part, p0, p1, ax, 0.010, 0.013, 0.010, Mat::MetalBare);
                        // The arm's outer face normal (the axle, square to the arm).
                        let dir = (p1 - p0).normalize();
                        let up = (ax - dir * ax.dot(dir)).normalize();
                        bar(
                            part,
                            p0 + up * 0.0105,
                            p1 + up * 0.0105,
                            up,
                            0.006,
                            0.009,
                            0.0008,
                            Mat::Chrome,
                        );
                    }
                    // Lug nuts.
                    let a = TAU * (k as f32 + 0.5) / 5.0;
                    let p = c + (e1 * a.cos() + e2 * a.sin()) * 0.050 + ax * (face + 0.010);
                    rounded_box(
                        part,
                        p,
                        Vec3::new(0.007, 0.007, 0.007),
                        0.003,
                        Mat3::IDENTITY,
                        Mat::Chrome,
                        2,
                    );
                }
            }
            // Brake disc and caliper.
            {
                let part = m.part(&format!("brake_disc_{tag}"));
                let w0 = -0.01;
                let prof: Vec<(f32, f32)> = vec![
                    (0.07, w0 + 0.014),
                    (0.165, w0 + 0.014),
                    (0.165, w0 - 0.014),
                    (0.07, w0 - 0.014),
                    (0.07, w0 + 0.014),
                ];
                revolve(part, c, ax, &prof, 64, Mat::MetalBare, false);
                let back = if front { 1.0 } else { -1.0 };
                let cp = c + Vec3::new(s * (w0 + 0.0), 0.10, -0.08 * back);
                let rot = Mat3::from_rotation_x(if front { -0.7 } else { 0.7 });
                rounded_box(
                    part,
                    cp,
                    Vec3::new(0.035, 0.045, 0.02),
                    0.012,
                    rot,
                    Mat::TrimBlack,
                    3,
                );
            }
        }
    }
}

fn mirrors_handles_wipers(m: &mut Model, door_f: &Grid, door_r: &Grid) {
    let _ = (door_f, door_r);
    for left in [true, false] {
        let s = if left { 1.0f32 } else { -1.0 };
        let side = if left { "L" } else { "R" };
        // Mirror on the sail: arm, housing (paint), glass (chrome).
        {
            let base = sf::hit(Param::Section, 0.66, sf::theta_at_y(0.66, 0.985));
            let base = Vec3::new(s * base.x, base.y, base.z);
            let part = m.part(&format!("mirror_{side}"));
            let hc = base + Vec3::new(s * 0.115, 0.045, -0.02);
            rounded_box(
                part,
                hc,
                Vec3::new(0.10, 0.055, 0.045),
                0.03,
                Mat3::from_rotation_y(s * 0.12),
                Mat::Paint,
                6,
            );
            rounded_box(
                part,
                hc + Vec3::new(0.0, 0.0, -0.046),
                Vec3::new(0.088, 0.045, 0.002),
                0.0015,
                Mat3::from_rotation_y(s * 0.12),
                Mat::Chrome,
                4,
            );
            bar(
                part,
                base + Vec3::new(s * 0.01, 0.0, 0.0),
                hc - Vec3::new(s * 0.05, 0.0, 0.0),
                Vec3::Y,
                0.016,
                0.014,
                0.012,
                Mat::TrimBlack,
            );
            m.marker(&format!("mount_mirror_{side}"), base);
        }
        // Door handles: flush, just under the shoulder line: a painted bar
        // standing 1.5 mm off the skin in a dark 3 mm reveal.
        for (front, z) in [(true, -0.20f32), (false, -0.90)] {
            let tag = if front { "F" } else { "R" };
            let y = sf::shoulder_y(z) - 0.032;
            let p = sf::hit(Param::Section, z, sf::theta_at_y(z, y));
            let n = sf::normal(p);
            let p = Vec3::new(s * p.x, p.y, p.z);
            let n = Vec3::new(s * n.x, n.y, n.z);
            let part = m.part(&format!("handle_{tag}{}", side));
            let side_v = Vec3::Y.cross(n).normalize();
            let rot = Mat3::from_cols(side_v, n.cross(side_v), n);
            rounded_box(
                part,
                p - n * 0.0015,
                Vec3::new(0.082, 0.0145, 0.0025),
                0.002,
                rot,
                Mat::TrimBlack,
                3,
            );
            rounded_box(
                part,
                p + n * 0.0002,
                Vec3::new(0.076, 0.0115, 0.0013),
                0.0012,
                rot,
                Mat::Paint,
                4,
            );
        }
        // Wipers on the cowl.
        {
            let part = m.part(&format!("wiper_{side}"));
            let x0: f32 = if left { 0.05 } else { -0.55 };
            let a = sf::hit(Param::Section, WS_BASE - 0.01, x0.atan2(0.4));
            let b = sf::hit(Param::Section, WS_BASE - 0.06, (x0 + 0.52).atan2(0.4));
            let n = sf::normal(a);
            bar(
                part,
                a + n * 0.012,
                b + n * 0.012,
                n,
                0.006,
                0.005,
                0.004,
                Mat::TrimBlack,
            );
        }
    }
}

fn interior(m: &mut Model) {
    let id = Mat3::IDENTITY;
    // Dash: a deep moulding under the windshield, its top padded.
    {
        let part = m.part("dash");
        rounded_box(
            part,
            Vec3::new(0.0, 0.80, 0.52),
            Vec3::new(0.70, 0.12, 0.22),
            0.07,
            id,
            Mat::InteriorPlastic,
            10,
        );
        rounded_box(
            part,
            Vec3::new(0.0, 0.62, 0.40),
            Vec3::new(0.66, 0.10, 0.14),
            0.05,
            id,
            Mat::InteriorPlastic,
            8,
        );
        // Instrument binnacle.
        rounded_box(
            part,
            Vec3::new(0.37, 0.93, 0.37),
            Vec3::new(0.17, 0.035, 0.08),
            0.03,
            id,
            Mat::InteriorPlastic,
            6,
        );
    }
    // Steering wheel (driver on the left, +X) and column.
    {
        let part = m.part("steering_wheel");
        let c = Vec3::new(0.37, 0.90, 0.17);
        let ax = Vec3::new(0.0, 0.42, -0.9).normalize();
        let tube: Vec<(f32, f32)> = (0..=16)
            .map(|q| {
                let a = TAU * q as f32 / 16.0;
                (0.185 + 0.016 * a.cos(), 0.016 * a.sin())
            })
            .collect();
        revolve(part, c, ax, &tube, 48, Mat::InteriorPlastic, true);
        rounded_box(
            part,
            c - ax * 0.02,
            Vec3::new(0.06, 0.06, 0.03),
            0.025,
            Mat3::from_rotation_x(-0.43),
            Mat::InteriorPlastic,
            4,
        );
        for a in [0.0f32, PI * 0.5, PI] {
            let e1 = ax.cross(Vec3::X).normalize();
            let e2 = ax.cross(e1);
            let d = e1 * a.cos() + e2 * a.sin();
            let _ = d;
        }
        let e1 = Vec3::X;
        bar(
            part,
            c - e1 * 0.17,
            c + e1 * 0.17,
            ax,
            0.018,
            0.018,
            0.008,
            Mat::InteriorPlastic,
        );
        bar(
            part,
            c - ax * 0.02,
            c + ax * 0.0 + Vec3::new(0.0, -0.17, -0.07),
            ax,
            0.02,
            0.02,
            0.008,
            Mat::InteriorPlastic,
        );
        bar(
            part,
            c - ax * 0.03,
            c - ax * 0.35,
            Vec3::Y,
            0.03,
            0.035,
            0.03,
            Mat::InteriorPlastic,
        );
    }
    // Seats.
    for (name, x) in [("seat_FL", 0.37f32), ("seat_FR", -0.37)] {
        let part = m.part(name);
        rounded_box(
            part,
            Vec3::new(x, 0.40, -0.40),
            Vec3::new(0.25, 0.06, 0.26),
            0.05,
            id,
            Mat::InteriorFabric,
            6,
        );
        let back = Mat3::from_rotation_x(-0.25);
        rounded_box(
            part,
            Vec3::new(x, 0.75, -0.72),
            Vec3::new(0.24, 0.30, 0.06),
            0.05,
            back,
            Mat::InteriorFabric,
            6,
        );
        rounded_box(
            part,
            Vec3::new(x, 1.12, -0.82),
            Vec3::new(0.12, 0.08, 0.05),
            0.04,
            back,
            Mat::InteriorFabric,
            4,
        );
        rounded_box(
            part,
            Vec3::new(x, 0.27, -0.40),
            Vec3::new(0.20, 0.07, 0.22),
            0.02,
            id,
            Mat::InteriorPlastic,
            3,
        );
    }
    {
        let part = m.part("seat_rear");
        rounded_box(
            part,
            Vec3::new(0.0, 0.40, -1.32),
            Vec3::new(0.66, 0.08, 0.25),
            0.06,
            id,
            Mat::InteriorFabric,
            8,
        );
        let back = Mat3::from_rotation_x(-0.32);
        rounded_box(
            part,
            Vec3::new(0.0, 0.76, -1.62),
            Vec3::new(0.66, 0.30, 0.07),
            0.06,
            back,
            Mat::InteriorFabric,
            8,
        );
        for x in [-0.42f32, 0.42] {
            rounded_box(
                part,
                Vec3::new(x, 1.10, -1.72),
                Vec3::new(0.11, 0.07, 0.05),
                0.04,
                back,
                Mat::InteriorFabric,
                4,
            );
        }
    }
    {
        let part = m.part("console");
        rounded_box(
            part,
            Vec3::new(0.0, 0.42, -0.10),
            Vec3::new(0.10, 0.14, 0.55),
            0.05,
            id,
            Mat::InteriorPlastic,
            6,
        );
        rounded_box(
            part,
            Vec3::new(0.0, 0.58, 0.18),
            Vec3::new(0.08, 0.08, 0.20),
            0.04,
            Mat3::from_rotation_x(0.5),
            Mat::InteriorPlastic,
            4,
        );
    }
    // Door trims: inside each door, from the floor to the belt.
    for (name, z0, z1) in [
        ("door_trim_FL", DOOR_F - 0.05, DOOR_B + 0.03),
        ("door_trim_RL", DOOR_B - 0.03, DOOR_R + 0.05),
    ] {
        let (a, b) = (
            Table::new(
                |u| sf::theta_at_y(u, sf::with_profiles(|p| p.belt.at(u)) - 0.01),
                -1.2,
                0.8,
            ),
            Table::new(|u| sf::theta_at_y(u, 0.36), -1.2, 0.8),
        );
        let mut st = Strip::new(
            Param::Section,
            k(z0),
            k(z1),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        st.off = -0.07;
        let g = grid(&st, 0.03).flipped();
        let right = name.replace('L', "R");
        emit(m.part(name), &g, Mat::InteriorPlastic, [1.0; 3]);
        emit(
            m.part(&right),
            &g.mirrored(),
            Mat::InteriorPlastic,
            [1.0; 3],
        );
    }
    // Headliner: the roof 30 mm in, facing down.
    {
        let a = Table::new(|u| sf::theta_corner(u) - 0.05, -1.2, 0.0);
        let b = a.clone();
        let mut st = Strip::new(
            Param::Section,
            k(WS_TOP - 0.03),
            k(BL_TOP + 0.03),
            Box::new(move |u| -a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        st.off = -0.03;
        let g = grid(&st, 0.03).flipped();
        emit(m.part("headliner"), &g, Mat::InteriorFabric, [1.0; 3]);
    }
    // Carpet (part of the console's floor): a flat floor inside.
    {
        let part = m.part("console");
        rounded_box(
            part,
            Vec3::new(0.0, 0.22, -0.55),
            Vec3::new(0.70, 0.01, 1.25),
            0.008,
            id,
            Mat::InteriorFabric,
            8,
        );
    }
}

fn bay(m: &mut Model) {
    let id = Mat3::IDENTITY;
    rounded_box(
        m.part("engine_block"),
        Vec3::new(0.0, 0.55, 1.60),
        Vec3::new(0.30, 0.22, 0.24),
        0.05,
        id,
        Mat::MetalBare,
        6,
    );
    rounded_box(
        m.part("engine_block"),
        Vec3::new(0.0, 0.79, 1.60),
        Vec3::new(0.26, 0.03, 0.10),
        0.02,
        id,
        Mat::TrimBlack,
        4,
    );
    rounded_box(
        m.part("radiator"),
        Vec3::new(0.0, 0.52, 2.12),
        Vec3::new(0.34, 0.18, 0.02),
        0.01,
        id,
        Mat::TrimBlack,
        6,
    );
    rounded_box(
        m.part("firewall"),
        Vec3::new(0.0, 0.60, 0.88),
        Vec3::new(0.70, 0.30, 0.006),
        0.004,
        id,
        Mat::MetalBare,
        10,
    );
    // Exhaust: a pipe under the floor splitting to two tips that end flush
    // in the diffuser, either side.
    let part = m.part("exhaust");
    let pipe: Vec<(f32, f32)> = vec![(0.0, 0.0), (0.03, 0.0), (0.03, 1.40), (0.0, 1.40)];
    revolve(
        part,
        Vec3::new(0.30, 0.20, -0.30),
        -Vec3::Z,
        &pipe,
        20,
        Mat::MetalBare,
        false,
    );
    for s in [-1.0f32, 1.0] {
        let (x, y) = (0.52f32, 0.315f32);
        let phi = phi_at_x(Param::Tail, x, y);
        let tip = sf::hit(Param::Tail, phi, y);
        let z1 = tip.z + 0.004;
        let z0 = -1.70;
        let l = z0 - z1;
        // A round tip: the pipe, a rolled 42 mm lip and its dark bore.
        let prof: Vec<(f32, f32)> = vec![
            (0.03, 0.0),
            (0.036, l - 0.10),
            (0.042, l - 0.01),
            (0.042, l),
            (0.036, l + 0.002),
            (0.033, l - 0.004),
            (0.033, l - 0.06),
        ];
        revolve(
            part,
            Vec3::new(s * x, y, z0),
            -Vec3::Z,
            &prof,
            28,
            Mat::Chrome,
            false,
        );
        let bore: Vec<(f32, f32)> = vec![(0.033, l - 0.06), (0.0, l - 0.06)];
        revolve(
            part,
            Vec3::new(s * x, y, z0),
            -Vec3::Z,
            &bore,
            28,
            Mat::TrimBlack,
            false,
        );
    }
}
fn markers(m: &mut Model) {
    for front in [true, false] {
        for left in [true, false] {
            let s = if left { 1.0f32 } else { -1.0 };
            let za = if front { AXLE_F } else { AXLE_R };
            let tag = wheel_name(front, left);
            m.markers.push(Marker {
                name: format!("wheel_center_{tag}"),
                pos: Vec3::new(s * TRACK_HALF, TYRE_R, za),
                // Local X along the axle, outward.
                rot: if left {
                    Quat::IDENTITY
                } else {
                    Quat::from_rotation_y(PI)
                },
            });
        }
    }
    let side_pt = |z: f32, y: f32| sf::hit(Param::Section, z, sf::theta_at_y(z, y));
    for left in [true, false] {
        let s = if left { 1.0f32 } else { -1.0 };
        let sx = |p: Vec3| Vec3::new(s * p.x, p.y, p.z);
        let side = if left { "L" } else { "R" };
        for (tag, zf, zr) in [("F", DOOR_F, DOOR_B), ("R", DOOR_B, DOOR_R)] {
            let name = format!("{tag}{side}");
            m.marker(
                &format!("hinge_door_{name}_upper"),
                sx(side_pt(zf - 0.03, 0.80)) - Vec3::new(s * 0.02, 0.0, 0.0),
            );
            m.marker(
                &format!("hinge_door_{name}_lower"),
                sx(side_pt(zf - 0.03, 0.45)) - Vec3::new(s * 0.02, 0.0, 0.0),
            );
            m.marker(
                &format!("latch_door_{name}"),
                sx(side_pt(zr, 0.70)) - Vec3::new(s * 0.03, 0.0, 0.0),
            );
        }
        m.marker(
            &format!("hinge_hood_{side}"),
            Vec3::new(s * 0.62, 0.92, HOOD_REAR + 0.06),
        );
        m.marker(
            &format!("hinge_trunk_{side}"),
            Vec3::new(s * 0.55, 0.99, BL_BASE - 0.05),
        );
        // Fender bolts: along the top flange, at the door and the bumper.
        let fl = format!("F{side}");
        let pts = [
            Vec3::new(0.80, 0.88, 1.0),
            Vec3::new(0.78, 0.84, 1.45),
            Vec3::new(0.70, 0.79, 1.95),
            sx(side_pt(DOOR_F + 0.04, 0.40)),
        ];
        for (q, p) in pts.iter().enumerate() {
            let p = if q < 3 {
                Vec3::new(s * p.x, p.y, p.z)
            } else {
                *p
            };
            m.marker(&format!("bolt_fender_{fl}_{}", q + 1), p);
        }
    }
    m.marker("latch_hood", Vec3::new(0.0, 0.70, 2.20));
    m.marker("latch_trunk", Vec3::new(0.0, 0.80, -2.42));
    for (q, x) in [-0.75f32, -0.35, 0.35, 0.75].iter().enumerate() {
        m.marker(
            &format!("bolt_bumper_front_{}", q + 1),
            Vec3::new(*x, 0.45, 2.24 - 0.12 * (x.abs() / 0.75).powi(2)),
        );
        m.marker(
            &format!("bolt_bumper_rear_{}", q + 1),
            Vec3::new(*x, 0.48, -2.38 + 0.10 * (x.abs() / 0.75).powi(2)),
        );
    }
    m.marker("seat_H_point_FL", Vec3::new(0.37, 0.50, -0.50));
    m.marker("seat_H_point_FR", Vec3::new(-0.37, 0.50, -0.50));
}

/// The model as the spec's glTF scene: UV0 packed into one non-overlapping
/// atlas at an even texel density, a primitive per material per part.
pub fn to_glb(m: &Model) -> crate::glb::Glb {
    // Shelf-pack every chart (metres, 2 cm apart).
    let pad = 0.02;
    let mut charts: Vec<(usize, usize, [f32; 2])> = Vec::new();
    for (pi, p) in m.parts.iter().enumerate() {
        for (ci, c) in p.charts.iter().enumerate() {
            charts.push((pi, ci, c.size));
        }
    }
    let area: f32 = charts.iter().map(|c| (c.2[0] + pad) * (c.2[1] + pad)).sum();
    let mut side = area.sqrt() * 1.1;
    let mut place: Vec<[f32; 2]>;
    let mut order: Vec<usize> = (0..charts.len()).collect();
    order.sort_by(|&a, &b| charts[b].2[1].total_cmp(&charts[a].2[1]));
    loop {
        place = vec![[0.0; 2]; charts.len()];
        let (mut x, mut y, mut row) = (0.0f32, 0.0f32, 0.0f32);
        let mut ok = true;
        for &k in &order {
            let s = charts[k].2;
            if x + s[0] + pad > side {
                x = 0.0;
                y += row + pad;
                row = 0.0;
            }
            if s[0] + pad > side {
                ok = false;
                break;
            }
            place[k] = [x, y];
            x += s[0] + pad;
            row = row.max(s[1]);
        }
        if ok && y + row <= side {
            break;
        }
        side *= 1.05;
    }
    let mut uv_of: Vec<Vec<[f32; 2]>> = m.parts.iter().map(|p| p.uv.clone()).collect();
    for (k, &(pi, ci, _)) in charts.iter().enumerate() {
        let c = &m.parts[pi].charts[ci];
        for v in c.first..c.first + c.count {
            let u = m.parts[pi].uv[v as usize];
            uv_of[pi][v as usize] = [(u[0] + place[k][0]) / side, (u[1] + place[k][1]) / side];
        }
    }
    let materials: Vec<crate::glb::Material> = Mat::ALL
        .iter()
        .map(|&mt| {
            let (base, metallic, roughness) = mt.pbr();
            crate::glb::Material {
                name: mt.name().into(),
                base,
                metallic,
                roughness,
                blend: base[3] < 1.0,
                emissive: if mt == Mat::LampEmissive {
                    [1.0, 0.95, 0.9]
                } else {
                    [0.0; 3]
                },
                // Paint: a metallic base under a glossy clearcoat.
                clearcoat: if mt == Mat::Paint { 1.0 } else { 0.0 },
                clearcoat_roughness: 0.03,
            }
        })
        .collect();
    let mut meshes = Vec::new();
    for (pi, p) in m.parts.iter().enumerate() {
        let mut mesh = crate::glb::Mesh {
            name: p.name.clone(),
            prims: Vec::new(),
        };
        let tinted = p.col.iter().any(|c| *c != [1.0; 3]);
        for (mi, &mt) in Mat::ALL.iter().enumerate() {
            let mut remap = vec![u32::MAX; p.pos.len()];
            let mut prim = crate::glb::Prim {
                material: mi,
                ..Default::default()
            };
            for (t, &tm) in p.tris.iter().zip(&p.tmat) {
                if tm != mt {
                    continue;
                }
                for &v in t {
                    if remap[v as usize] == u32::MAX {
                        remap[v as usize] = prim.pos.len() as u32;
                        prim.pos.push(p.pos[v as usize].to_array());
                        prim.nrm
                            .push(p.nrm[v as usize].normalize_or(Vec3::Y).to_array());
                        prim.uv.push(uv_of[pi][v as usize]);
                        if tinted {
                            prim.col.push(p.col[v as usize]);
                        }
                    }
                    prim.idx.push(remap[v as usize]);
                }
            }
            if !prim.idx.is_empty() {
                mesh.prims.push(prim);
            }
        }
        if !mesh.prims.is_empty() {
            meshes.push(mesh);
        }
    }
    crate::glb::Glb {
        root: "vehicle".into(),
        meshes,
        materials,
        markers: m
            .markers
            .iter()
            .map(|k| crate::glb::Marker {
                name: k.name.clone(),
                pos: k.pos.to_array(),
                rot: k.rot.to_array(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_every_part() {
        let m = build();
        for name in [
            "body_shell",
            "hood",
            "trunk_lid",
            "door_FL",
            "door_RR",
            "fender_FL",
            "bumper_front",
            "grille",
            "glass_windshield",
            "headlight_L_lens",
            "taillight_R_housing",
            "rim_FL",
            "tyre_RR",
            "seat_FL",
        ] {
            assert!(
                m.parts.iter().any(|p| p.name == name && !p.tris.is_empty()),
                "{name}"
            );
        }
    }
}
