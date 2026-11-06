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
/// Hem roll radius and flange depth (m): the skin rolls over 7 mm.
pub const HEM_R: f32 = 0.00225;
pub const HEM_FLANGE: f32 = 0.0055;

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
            Mat::Paint => ([0.42, 0.035, 0.03, 1.0], 0.0, 0.28),
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
    /// A character line across the strip (its b as a function of a): one
    /// grid row runs exactly along it (an edge loop on the crease).
    pub crease: Option<F>,
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
            crease: None,
        }
    }
    /// The t of the crease at s (None without one).
    fn crease_t(&self, s: f32) -> Option<f32> {
        let c = self.crease.as_ref()?;
        let mut t = 0.5;
        for _ in 0..8 {
            let a = (self.a0)(t) + ((self.a1)(t) - (self.a0)(t)) * s;
            let (lo, hi) = ((self.lo)(a), (self.hi)(a));
            t = ((c(a) - lo) / (hi - lo)).clamp(0.08, 0.92);
        }
        Some(t)
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
    // A crease row: j_c, pinned to the crease; the columns even out on
    // either side of it.
    let nb = if st.crease.is_some() { nb.max(2) } else { nb };
    let jc = st.crease.as_ref().map(|_| {
        let mean = (0..=8)
            .filter_map(|q| st.crease_t(q as f32 / 8.0))
            .sum::<f32>()
            / 9.0;
        ((mean * nb as f32).round() as usize).clamp(1, nb - 1)
    });
    let mut st_ij: Vec<(f32, f32)> = (0..=nb)
        .flat_map(|j| {
            (0..=ns).map(move |i| {
                let s = i as f32 / ns as f32;
                let t = match (jc, st.crease_t(s)) {
                    (Some(jc), Some(tc)) if j <= jc => tc * j as f32 / jc as f32,
                    (Some(jc), Some(tc)) => tc + (1.0 - tc) * (j - jc) as f32 / (nb - jc) as f32,
                    _ => j as f32 / nb as f32,
                };
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
                if Some(j) == jc {
                    st_ij[at(i, j)].1 = st.crease_t(e[i].0).unwrap();
                }
            }
        }
        // Columns (in two runs either side of a crease row).
        let runs: Vec<(usize, usize)> = match jc {
            Some(c) => vec![(0, c), (c, nb)],
            None => vec![(0, nb)],
        };
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

/// Append a grid to a part (one UV chart); returns its vertex indices.
pub fn emit(part: &mut Part, g: &Grid, m: Mat, col: [f32; 3]) -> Vec<u32> {
    let first = part.pos.len() as u32;
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
    let ids: Vec<u32> = (0..g.p.len()).map(|k| raw[root(&mut rep, k)]).collect();
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
                3 => part.tri([u[0], u[1], u[2]], m),
                _ => {}
            }
        }
    }
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
/// deep, then a flange [`HEM_FLANGE`] back under the skin.
pub fn hem(part: &mut Part, g: &Grid, ids: &[u32], e: usize, m: Mat) {
    let idx = g.edge(e);
    let inner: Vec<usize> = match e {
        0 => (0..=g.ns).map(|i| g.at(i, 1.min(g.nb))).collect(),
        1 => (0..=g.ns)
            .map(|i| g.at(i, g.nb.saturating_sub(1)))
            .collect(),
        2 => (0..=g.nb).map(|j| g.at(1.min(g.ns), j)).collect(),
        _ => (0..=g.nb)
            .map(|j| g.at(g.ns.saturating_sub(1), j))
            .collect(),
    };
    let mut prev: Vec<u32> = idx.iter().map(|&k| ids[k]).collect();
    let rows: Vec<Vec<(Vec3, Vec3)>> = {
        let mut rows = vec![Vec::new(); 2];
        for (q, &k) in idx.iter().enumerate() {
            let p = g.p[k];
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
            rows[0].push((p - n * (2.0 * HEM_R), (o - n * 0.4).normalize()));
            rows[1].push((p - n * (2.0 * HEM_R) - o * HEM_FLANGE, -n));
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
    smooth_profile: bool,
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
        let ring: Vec<u32> = (0..seg)
            .map(|s| {
                let a = TAU * s as f32 / seg as f32;
                let rad = e1 * a.cos() + e2 * a.sin();
                let n = if smooth_profile {
                    (rad * nr + ax * nw).normalize_or(rad)
                } else {
                    (rad * nr + ax * nw).normalize_or(rad)
                };
                part.vert(
                    c + rad * r + ax * w,
                    n,
                    [s as f32 / seg as f32, k as f32 / np as f32],
                    [1.0; 3],
                )
            })
            .collect();
        rings.push(ring);
    }
    for k in 0..np - 1 {
        for s in 0..seg {
            let s2 = (s + 1) % seg;
            part.quad(
                rings[k][s],
                rings[k][s2],
                rings[k + 1][s2],
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
const BL_TOP: f32 = -1.05;
const BL_BASE: f32 = -1.80;
/// The hood's rear edge (the cowl between it and the windshield).
const HOOD_REAR: f32 = 0.86;
/// Bumper top on the sides (front, rear).
const BUMPER_TOP_F: f32 = 0.625;
const BUMPER_TOP_R: f32 = 0.80;

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
    let th_corner = Table::new(sf::theta_corner, -1.9, 0.85);
    let th_top_edge = {
        let c = th_corner.clone();
        Table::new(move |u| theta_walk(u, c.at(u), -0.05), -1.9, 0.85)
    };
    let th_frame_top = {
        let c = th_corner.clone();
        let b = th_belt.clone();
        Table::new(
            move |u| theta_walk(u, c.at(u), 0.035).min(b.at(u) - 0.02),
            -1.9,
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
    let x_blb = 0.70;
    let th_bl_edge = Table::new(
        move |u| {
            sf::theta_at_x(
                u,
                x_bl_top + (x_blb - x_bl_top) * sf::smoothstep(BL_TOP, BL_BASE, u),
            )
        },
        -1.9,
        -0.9,
    );
    let x_trunk = move |u: f32| {
        let w = x_blb + (0.76 - x_blb) * sf::smoothstep(BL_BASE, BL_BASE - 0.30, u);
        w - 0.20 * sf::smoothstep(-1.95, -2.42, u).powf(1.4)
    };
    let th_trunk = Table::new(move |u| sf::theta_at_x(u, x_trunk(u)), -2.5, -1.6);

    // ------------------------------------------------------------------
    // Hood: from the cowl to its leading edge (bowed back to the corners).
    // ------------------------------------------------------------------
    let hood_front = |t: f32| 2.115 + 0.10 * (1.0 - (2.0 * t - 1.0).powi(2));
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
            ZC_F + (2.115 - ZC_F) * w.powf(1.6)
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
        s.crease = Some(Box::new(move |u| c.at(u)));
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
        st.crease = Some(Box::new(move |u| c.at(u)));
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
    let dlo_rear_belt = -1.40;
    let dlo_rear_top = -1.03;
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
        Table::new(move |u| theta_walk(u, f.at(u), frame_w), -1.9, 0.85)
    };
    let th_glass_bottom = {
        let b = th_belt.clone();
        Table::new(move |u| theta_walk(u, b.at(u), -0.012), -1.9, 0.85)
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
        [g1, g2, z0, g2],
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
            [false, true, false, false],
            [false, true, false, false],
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
    let quarter_end = -2.32;
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
            [g1, z0, z0, z0],
            [true, false, false, false],
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
        panel_by(
            &mut m,
            "body_shell",
            Some("body_shell"),
            &st,
            Mat::Paint,
            [g1, z0, z0, g1],
            [true, false, false, true],
        );
    }
    // Quarter below the belt: rear door to the tail lamp (over the bumper).
    let t_seam_r = {
        let lo = th_belt.at(ZC_R);
        let hi = th_side_bottom.at(ZC_R);
        (sf::theta_at_y(ZC_R, sf::shoulder_y(ZC_R)) - lo) / (hi - lo)
    };
    let quarter_rear = move |t: f32| {
        if t >= t_seam_r {
            ZC_R
        } else {
            let w = 1.0 - t / t_seam_r;
            ZC_R + (quarter_end - ZC_R) * w.powf(1.3)
        }
    };
    let quarter = {
        let (a, b) = (th_belt.clone(), th_side_bottom.clone());
        let mut s = Strip::new(
            Param::Section,
            k(DOOR_R),
            Box::new(quarter_rear),
            Box::new(move |u| a.at(u)),
            Box::new(move |u| b.at(u)),
        );
        let c = th_shoulder.clone();
        s.crease = Some(Box::new(move |u| c.at(u)));
        s
    };
    let g_quarter = panel_by(
        &mut m,
        "body_shell",
        Some("body_shell"),
        &quarter,
        Mat::Paint,
        [0.0, GAP, GAP, GAP],
        [false, true, true, true],
    );
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
    let deck_rear = |t: f32| -2.375 - 0.075 * (1.0 - (2.0 * t - 1.0).powi(2));
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
    // The rear face in the tail family.
    let fold = g_trunk.edge_points(3);
    let fold_y = phi_y(Param::Tail, &fold);
    let x_tl_in = 0.47;
    let phi_tl_in = phi_at_x(Param::Tail, x_tl_in, 0.80);
    let y_rbt = |phi: f32| BUMPER_TOP_R - 0.015 * (1.0 - (phi / FRAC_PI_2).powi(2)).max(0.0) + 0.0;
    {
        let fy = phi_y(Param::Tail, &fold);
        let st = Strip::new(
            Param::Tail,
            k(-phi_tl_in),
            k(phi_tl_in),
            Box::new(y_rbt),
            fy,
        );
        let mut g = grid(&st, H);
        g.shrink([true, false, true, true], GAP);
        let part = m.part("trunk_lid");
        let ids = emit(part, &g, Mat::Paint, [1.0; 3]);
        for e in [0, 2, 3] {
            hem(part, &g, &ids, e, Mat::Paint);
        }
    }
    let _ = fold_y;
    // Tail lamps: the rear face's corners up to the quarter's and the deck's
    // edges, wrapping to the quarter's tail.
    {
        let mut top = Vec::new();
        top.extend(fold.iter().copied().filter(|p| p.x > 0.0));
        let qe = g_quarter.edge_points(3);
        top.extend(qe.iter().copied());
        // The C-pillar strip's tail end.
        let fy = phi_y(Param::Tail, &top);
        let phi_end = FRAC_PI_2 - 0.01;
        let st = Strip::new(Param::Tail, k(phi_tl_in), k(phi_end), Box::new(y_rbt), fy);
        let mut g = grid(&st, H);
        g.shrink([true, true, true, true], GAP);
        lamp(&mut m, "taillight_L", &g, [0.75, 0.03, 0.03]);
        lamp(&mut m, "taillight_R", &g.mirrored(), [0.75, 0.03, 0.03]);
    }

    // ------------------------------------------------------------------
    // Nose: grille, headlamps, bumper.
    // ------------------------------------------------------------------
    let hood_edge = g_hood.edge_points(3);
    let fender_edge = g_fender.edge_points(3);
    let x_grille = 0.36;
    let phi_g = phi_at_x(Param::Nose, x_grille, 0.62);
    let y_bt_f = move |phi: f32| {
        let w = sf::smoothstep(phi_g - 0.02, phi_g + 0.08, phi.abs());
        0.545 + (BUMPER_TOP_F - 0.545) * w
    };
    {
        let mut top: Vec<Vec3> = hood_edge.iter().copied().filter(|p| p.x > 0.0).collect();
        top.extend(fender_edge.iter().copied());
        let fy = phi_y(Param::Nose, &top);
        let st = Strip::new(
            Param::Nose,
            k(phi_g),
            k(FRAC_PI_2 - 0.01),
            Box::new(y_bt_f),
            fy,
        );
        let mut g = grid(&st, H);
        g.shrink([true, true, true, true], GAP);
        lamp(&mut m, "headlight_L", &g, [1.0; 3]);
        lamp(&mut m, "headlight_R", &g.mirrored(), [1.0; 3]);
    }
    // Grille: a chrome surround round a recessed black insert with bars.
    {
        let fy = phi_y(Param::Nose, &hood_edge);
        let st = Strip::new(Param::Nose, k(-phi_g), k(phi_g), Box::new(y_bt_f), fy);
        let mut g = grid(&st, 0.018);
        g.shrink([true, true, true, true], GAP);
        grille(&mut m, &g);
    }
    // Bumper: under everything, round to the arches; a lower intake.
    let y_bb = 0.30;
    let phi_i = phi_at_x(Param::Nose, 0.50, 0.40);
    let (yi0, yi1) = (0.33, 0.45);
    {
        // Under the intake, over it, and the two sides (to the arches).
        let pieces: [(f32, f32, F, F, [bool; 4]); 4] = [
            (-phi_i, phi_i, k(y_bb), k(yi0), [true, false, false, false]),
            (
                -phi_i,
                phi_i,
                k(yi1),
                Box::new(y_bt_f),
                [false, true, false, false],
            ),
            (
                phi_i,
                FRAC_PI_2,
                arch_lo(Param::Nose, y_bb),
                Box::new(y_bt_f),
                [true, true, false, true],
            ),
            (
                -FRAC_PI_2,
                -phi_i,
                arch_lo(Param::Nose, y_bb),
                Box::new(y_bt_f),
                [true, true, true, false],
            ),
        ];
        for (a0, a1, lo, hi, gaps) in pieces {
            let st = Strip::new(Param::Nose, k(a0), k(a1), lo, hi);
            let mut g = grid(&st, H);
            g.shrink(gaps, GAP);
            let part = m.part("bumper_front");
            let ids = emit(part, &g, Mat::Paint, [1.0; 3]);
            for e in 0..4 {
                if gaps[e] {
                    hem(part, &g, &ids, e, Mat::Paint);
                }
            }
        }
        // Intake insert (recessed black mesh).
        let st = Strip::new(Param::Nose, k(-phi_i), k(phi_i), k(yi0), k(yi1));
        let g = grid(&st, H);
        recess(
            &mut m,
            "bumper_front",
            &g,
            0.04,
            Mat::TrimBlack,
            Mat::TrimBlack,
        );
    }
    // Rear bumper.
    {
        let st = Strip::new(
            Param::Tail,
            k(-FRAC_PI_2),
            k(FRAC_PI_2),
            arch_lo(Param::Tail, 0.34),
            Box::new(y_rbt),
        );
        let mut g = grid(&st, H);
        g.shrink([true, true, true, true], GAP);
        let part = m.part("bumper_rear");
        let ids = emit(part, &g, Mat::Paint, [1.0; 3]);
        for e in 0..4 {
            hem(part, &g, &ids, e, Mat::Paint);
        }
        // A black diffuser band along the bottom.
        let st = Strip::new(Param::Tail, k(-1.2), k(1.2), Box::new(|_| 0.30), k(0.34));
        let g = grid(&st, H).offset(0.002);
        emit(m.part("bumper_rear"), &g, Mat::TrimBlack, [1.0; 3]);
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

/// A nose or tail panel's lower edge: `base`, rising (to the wheel centre's
/// height at the sides) and following the wheel arch where the panel
/// reaches it.
fn arch_lo(param: Param, base: f32) -> F {
    tab(
        move |phi| {
            let z = sf::hit(param, phi, 0.45).z;
            let b = base + (TYRE_R - base).max(0.0) * sf::smoothstep(1.05, 1.35, phi.abs());
            b.max(sf::arch_y(z).unwrap_or(0.0))
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

/// A lamp: its lens on the surface, its housing behind (a chrome reflector
/// 50 mm back, the throat between, a light guide along the top).
fn lamp(m: &mut Model, name: &str, g: &Grid, tint: [f32; 3]) {
    let lens = g.offset(0.0015);
    emit(m.part(&format!("{name}_lens")), &lens, Mat::LampLens, tint);
    let depth = 0.05;
    let back = g.offset(-depth).flipped();
    let hp = format!("{name}_housing");
    let red = tint[1] < 0.5;
    emit(
        m.part(&hp),
        &back,
        if red { Mat::TrimBlack } else { Mat::Chrome },
        [1.0; 3],
    );
    throat(m.part(&hp), g, depth, Mat::TrimBlack);
    // Light guide: a band in the lens's upper third, 15 mm behind it.
    let (j0, j1) = ((g.nb * 2) / 3, ((g.nb * 2) / 3 + 2).min(g.nb));
    let mut band = Part::new("");
    let _ = &mut band;
    let part = m.part(&hp);
    let mut ids = Vec::new();
    for j in [j0, j1] {
        let row: Vec<u32> = (0..=g.ns)
            .map(|i| {
                let k = g.at(i, j);
                part.vert(g.p[k] - g.n[k] * 0.015, g.n[k], [0.0; 2], tint)
            })
            .collect();
        ids.push(row);
    }
    for i in 0..g.ns {
        part.quad(
            ids[0][i],
            ids[0][i + 1],
            ids[1][i + 1],
            ids[1][i],
            Mat::LampEmissive,
        );
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

/// The grille: a chrome frame round the opening, a recessed black insert,
/// two chrome bars across.
fn grille(m: &mut Model, g: &Grid) {
    let part = m.part("grille");
    // Frame: the outer two rings of cells.
    let ring = 1usize.min(g.ns / 2).min(g.nb / 2);
    let vid: Vec<u32> = (0..g.p.len())
        .map(|k| part.vert(g.p[k] + g.n[k] * 0.002, g.n[k], g.uv[k], [1.0; 3]))
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
    // Inner opening.
    let inner = Grid {
        ns: g.ns - 2 * ring,
        nb: g.nb - 2 * ring,
        p: (ring..=g.nb - ring)
            .flat_map(|j| (ring..=g.ns - ring).map(move |i| (i, j)))
            .map(|(i, j)| g.p[g.at(i, j)] + g.n[g.at(i, j)] * 0.002)
            .collect(),
        n: (ring..=g.nb - ring)
            .flat_map(|j| (ring..=g.ns - ring).map(move |i| (i, j)))
            .map(|(i, j)| g.n[g.at(i, j)])
            .collect(),
        uv: (ring..=g.nb - ring)
            .flat_map(|j| (ring..=g.ns - ring).map(move |i| (i, j)))
            .map(|(i, j)| g.uv[g.at(i, j)])
            .collect(),
    };
    let depth = 0.035;
    emit(part, &inner.offset(-depth), Mat::TrimBlack, [1.0; 3]);
    throat(part, &inner, depth, Mat::TrimBlack);
    // Bars.
    for frac in [0.36f32, 0.68] {
        let j = ((inner.nb as f32 * frac).round() as usize).min(inner.nb);
        let a = inner.p[inner.at(0, j)] - inner.n[inner.at(0, j)] * 0.012;
        let b = inner.p[inner.at(inner.ns, j)] - inner.n[inner.at(inner.ns, j)] * 0.012;
        let mid = inner.p[inner.at(inner.ns / 2, j)] - inner.n[inner.at(inner.ns / 2, j)] * 0.012;
        bar(part, a, mid, Vec3::Y, 0.006, 0.006, 0.004, Mat::Chrome);
        bar(part, mid, b, Vec3::Y, 0.006, 0.006, 0.004, Mat::Chrome);
    }
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
                let prof: Vec<(f32, f32)> = prof.into_iter().map(|(r, w)| (r, w)).collect();
                revolve(part, c, ax, &prof, 72, Mat::Rubber, true);
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
                // Hub disc and centre cap.
                let face = hw - 0.035;
                let hub: Vec<(f32, f32)> = vec![
                    (0.0, face + 0.012),
                    (0.035, face + 0.012),
                    (0.05, face + 0.006),
                    (0.075, face),
                    (0.085, face - 0.02),
                ];
                revolve(part, c, ax, &hub, 48, Mat::MetalBare, true);
                let cap: Vec<(f32, f32)> = vec![(0.0, face + 0.0125), (0.03, face + 0.0125)];
                let _ = cap;
                // Spokes: five pairs, from the hub to the lip, dished.
                let e1 = Vec3::Y;
                let e2 = ax.cross(e1);
                for k in 0..5 {
                    for d in [-0.11f32, 0.11] {
                        let a = TAU * k as f32 / 5.0 + d;
                        let dir = e1 * a.cos() + e2 * a.sin();
                        let p0 = c + dir * 0.065 + ax * (face - 0.002);
                        let p1 = c + dir * (RIM_R - 0.018) + ax * (hw - 0.03);
                        bar(part, p0, p1, ax, 0.012, 0.018, 0.011, Mat::MetalBare);
                    }
                    // Lug nuts.
                    let a = TAU * (k as f32 + 0.5) / 5.0;
                    let p = c + (e1 * a.cos() + e2 * a.sin()) * 0.058 + ax * (face + 0.012);
                    rounded_box(
                        part,
                        p,
                        Vec3::new(0.008, 0.008, 0.008),
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
        // Door handles: proud of the skin on the shoulder line.
        for (front, z) in [(true, -0.22f32), (false, -0.88)] {
            let tag = if front { "F" } else { "R" };
            let y = 0.885;
            let p = sf::hit(Param::Section, z, sf::theta_at_y(z, y));
            let n = sf::normal(p);
            let p = Vec3::new(s * p.x, p.y, p.z);
            let n = Vec3::new(s * n.x, n.y, n.z);
            let part = m.part(&format!("handle_{tag}{}", side));
            let rot = Mat3::from_cols(
                Vec3::Z.cross(n).normalize() * 1.0,
                n.cross(Vec3::Z.cross(n).normalize()),
                n,
            );
            let rot = Mat3::from_cols(Vec3::Z, Vec3::Y, Vec3::Z.cross(Vec3::Y)) * Mat3::IDENTITY;
            let _ = rot;
            let side_v = Vec3::Y.cross(n).normalize();
            let rot = Mat3::from_cols(side_v, n.cross(side_v), n);
            rounded_box(
                part,
                p + n * 0.008,
                Vec3::new(0.085, 0.014, 0.008),
                0.007,
                rot,
                Mat::Chrome,
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
    // Exhaust: a pipe under the floor to a tip under the rear bumper.
    let part = m.part("exhaust");
    let prof: Vec<(f32, f32)> = vec![
        (0.0, 0.0),
        (0.03, 0.0),
        (0.03, 2.06),
        (0.034, 2.06),
        (0.034, 2.12),
        (0.0, 2.12),
    ];
    revolve(
        part,
        Vec3::new(0.40, 0.20, -0.30),
        -Vec3::Z,
        &prof,
        20,
        Mat::MetalBare,
        false,
    );
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
                sx(side_pt(zr + 0.03, 0.70)) - Vec3::new(s * 0.03, 0.0, 0.0),
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
            sx(side_pt(DOOR_F + 0.04, 0.40)) * Vec3::new(s, 1.0, 1.0),
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
