//! The visual car as constructed parts: a dense body surface cut into separate
//! panels along smooth shut lines, and modelled fixtures (lamps, grille, glass,
//! mirrors, handles, wheels, brakes, arch liners, underbody, exhaust).
//!
//! Everything is generated here from the definition (`VehicleDef`): the body's
//! package curves ([`BodyProfile`](super::body::BodyProfile)) and a styling
//! description ([`design::Styling`]). Nothing depends on the skeleton's node
//! spacing: [`super::skin`] binds every piece to the solver's triangles
//! afterwards, so the same mesh rides a coarse or a dense skeleton.
//!
//! * [`body`]: the body surface: a lower *band* swept round the plan outline
//!   (bumpers, sides, the tail) and a *lid* over it (bonnet, greenhouse, deck),
//!   one connected triangle mesh with smooth section curves, a shoulder crease,
//!   arch flares and tumblehome;
//! * [`trim`]: cutting that surface into panels: regions as signed distance
//!   functions, marching triangles, rolled panel edges with a real gap and a
//!   return flange, offset copies (glass, inner skins);
//! * [`fixtures`]: the parametric part tools (lamp, grille, slats, mirror,
//!   handle, wheel, brake, arch liner, exhaust);
//! * [`design`]: the sedan's styling numbers and [`design::assemble`], which
//!   lays the parts out.
//!
//! Guide: `engine/docs/vehicle-authoring.md` ("The visual car").

pub mod body;
pub mod design;
pub mod fixtures;
pub mod trim;

use glam::Vec3;
use std::cell::Cell;

thread_local! {
    static DETAIL: Cell<f32> = const { Cell::new(1.0) };
}

/// The level of detail being built (1 full; lower for the distant LOD): the
/// tools scale their segment counts and edge rolls by it.
pub fn detail() -> f32 {
    DETAIL.with(|d| d.get())
}

/// Run `f` building at level of detail `d`.
pub fn with_detail<T>(d: f32, f: impl FnOnce() -> T) -> T {
    let old = DETAIL.with(|c| c.replace(d));
    let r = f();
    DETAIL.with(|c| c.set(old));
    r
}

/// A segment count scaled by the level of detail (at least `min`).
pub fn segs(n: usize, min: usize) -> usize {
    ((n as f32 * detail()).round() as usize).max(min)
}

/// How a piece rides the skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    /// The physics part its vertices lie on (a majority vote of the nearest skin
    /// triangles), then every vertex on that part's triangles.
    Vote,
    /// Each vertex on the nearest skin triangle of any part.
    Nearest,
    /// A given physics part (a voted piece after its cut along part seams).
    Part(u16),
    /// A named structure group (seats on their frames).
    Group(u16),
    /// A rigid part (mirror, handle, lamp): voted part, then the whole piece
    /// on one triangle's orthonormal frame (moves rigidly, never stretches).
    Rigid,
    /// A wheel: the tyre's fan triangles (rotates with it), by tyre index.
    Wheel(u16),
    /// A wheel's non-rotating hardware (brake caliper): axle nodes and a knuckle node.
    Hub(u16),
}

/// A triangle mesh with per-vertex material and ambient occlusion.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub pos: Vec<Vec3>,
    pub nrm: Vec<Vec3>,
    pub mat: Vec<u8>,
    /// Ambient occlusion (1 open, 0 fully occluded).
    pub ao: Vec<f32>,
    pub idx: Vec<u32>,
    /// 1 where the normal follows the skin's node normals when deformed (smooth
    /// body surfaces); 0 rides its binding triangle's frame.
    pub smooth: Vec<u8>,
}

impl Mesh {
    pub fn vert(&mut self, p: Vec3, n: Vec3, mat: u8, ao: f32) -> u32 {
        self.pos.push(p);
        self.nrm.push(n.normalize_or_zero());
        self.mat.push(mat);
        self.ao.push(ao);
        self.smooth.push(0);
        (self.pos.len() - 1) as u32
    }

    pub fn tri(&mut self, a: u32, b: u32, c: u32) {
        if a != b && b != c && a != c {
            self.idx.extend_from_slice(&[a, b, c]);
        }
    }

    /// A quad a b c d (in order round it).
    pub fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.tri(a, b, c);
        self.tri(a, c, d);
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    pub fn append(&mut self, o: &Mesh) {
        let base = self.pos.len() as u32;
        self.pos.extend_from_slice(&o.pos);
        self.nrm.extend_from_slice(&o.nrm);
        self.mat.extend_from_slice(&o.mat);
        self.ao.extend_from_slice(&o.ao);
        self.smooth.extend_from_slice(&o.smooth);
        self.idx.extend(o.idx.iter().map(|i| i + base));
    }

    /// Strips of rings: `rings[k]` is a ring of vertex ids; quads join ring k to
    /// k + 1 (closed rings when `closed`).
    pub fn strips(&mut self, rings: &[Vec<u32>], closed: bool) {
        for w in rings.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let n = a.len().min(b.len());
            let m = if closed { n } else { n - 1 };
            for i in 0..m {
                let j = (i + 1) % n;
                self.quad(a[i], a[j], b[j], b[i]);
            }
        }
    }

    /// Make every triangle face the side `outward` says (a point → a direction
    /// the face should look toward).
    pub fn orient(&mut self, outward: impl Fn(Vec3) -> Vec3) {
        for t in self.idx.chunks_mut(3) {
            let (a, b, c) = (
                self.pos[t[0] as usize],
                self.pos[t[1] as usize],
                self.pos[t[2] as usize],
            );
            let n = (b - a).cross(c - a);
            if n.dot(outward((a + b + c) / 3.0)) < 0.0 {
                t.swap(1, 2);
            }
        }
    }

    /// Wind every triangle to face its vertices' normals (so the renderer can
    /// tell a panel's outside from its inside).
    pub fn orient_by_normals(&mut self) {
        for t in self.idx.chunks_mut(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let f = (self.pos[b] - self.pos[a]).cross(self.pos[c] - self.pos[a]);
            if f.dot(self.nrm[a] + self.nrm[b] + self.nrm[c]) < 0.0 {
                t.swap(1, 2);
            }
        }
    }

    /// Smooth vertex normals from the faces (area weighted).
    pub fn face_normals(&mut self) {
        let mut acc = vec![Vec3::ZERO; self.pos.len()];
        for t in self.idx.chunks(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let n = (self.pos[b] - self.pos[a]).cross(self.pos[c] - self.pos[a]);
            acc[a] += n;
            acc[b] += n;
            acc[c] += n;
        }
        for (n, a) in self.nrm.iter_mut().zip(acc) {
            if a.length_squared() > 1e-20 {
                *n = a.normalize();
            }
        }
    }

    /// A surface of revolution: `profile` is (radius, axial offset) pairs swept
    /// round `axis` about `centre` with `segs` segments; `s0` the zero angle
    /// direction. Normals from the profile's tangent. Returns the rings.
    #[allow(clippy::too_many_arguments)]
    pub fn lathe(
        &mut self,
        centre: Vec3,
        axis: Vec3,
        s0: Vec3,
        profile: &[(f32, f32)],
        segs: usize,
        mat: impl Fn(usize, f32) -> u8,
        ao: impl Fn(usize) -> f32,
        sharp: bool,
    ) -> Vec<Vec<u32>> {
        let s1 = axis.cross(s0);
        let mut rings = Vec::new();
        let n = profile.len();
        // Sharp: every profile segment gets its own (flat-normal) pair of rings.
        let spans: Vec<(usize, usize)> = if sharp {
            (0..n - 1).map(|k| (k, k + 1)).collect()
        } else {
            vec![(0, n - 1)]
        };
        for (lo, hi) in spans {
            let mut seg_rings = Vec::new();
            for k in lo..=hi {
                let (r, w) = profile[k];
                let (ka, kb) = if sharp {
                    (lo, hi)
                } else {
                    (k.saturating_sub(1), (k + 1).min(n - 1))
                };
                let tr = profile[kb].0 - profile[ka].0;
                let tw = profile[kb].1 - profile[ka].1;
                let mut ring = Vec::with_capacity(segs);
                for i in 0..segs {
                    let a = std::f32::consts::TAU * i as f32 / segs as f32;
                    let d = s0 * a.cos() + s1 * a.sin();
                    // Tangent along the profile: d·tr + axis·tw; the normal is
                    // perpendicular to it within the (d, axis) plane.
                    let nn = d * tw - axis * tr;
                    let nn = if nn.length_squared() < 1e-12 {
                        axis * w.signum()
                    } else {
                        nn
                    };
                    let p = centre + d * r + axis * w;
                    ring.push(self.vert(p, nn, mat(k, a), ao(k)));
                }
                seg_rings.push(ring);
            }
            self.strips(&seg_rings, true);
            rings.extend(seg_rings);
        }
        rings
    }

    /// A box with rounded-off look (normals bent toward the corners), centre `c`,
    /// half extents along the frame (u, v, w).
    #[allow(clippy::too_many_arguments)]
    pub fn block(&mut self, c: Vec3, u: Vec3, v: Vec3, w: Vec3, h: Vec3, mat: u8, ao: f32) {
        let axes = [u, v, w];
        for ax in 0..3 {
            for s in [-1.0f32, 1.0] {
                let n = axes[ax] * s;
                let a = axes[(ax + 1) % 3];
                let b = axes[(ax + 2) % 3];
                let (ha, hb, hn) = (h[(ax + 1) % 3], h[(ax + 2) % 3], h[ax]);
                let mut ids = [0u32; 4];
                for (k, (x, y)) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                    .iter()
                    .enumerate()
                {
                    let p = c + n * hn + a * (x * ha) + b * (y * hb);
                    let nn = n + (a * *x + b * *y) * 0.2;
                    ids[k] = self.vert(p, nn, mat, ao);
                }
                let (pa, pb, pc) = (
                    self.pos[ids[0] as usize],
                    self.pos[ids[1] as usize],
                    self.pos[ids[2] as usize],
                );
                if (pb - pa).cross(pc - pa).dot(n) < 0.0 {
                    self.quad(ids[0], ids[3], ids[2], ids[1]);
                } else {
                    self.quad(ids[0], ids[1], ids[2], ids[3]);
                }
            }
        }
    }

    /// A tube along a polyline `path` with a superellipse section (half sizes
    /// `hw` across `side`, `hh` along the path's up), `segs` round.
    pub fn tube(&mut self, path: &[Vec3], up: Vec3, hw: f32, hh: f32, segs: usize, mat: u8, ao: f32) {
        let mut rings = Vec::new();
        for (k, &p) in path.iter().enumerate() {
            let t = (path[(k + 1).min(path.len() - 1)] - path[k.saturating_sub(1)]).normalize_or_zero();
            let side = t.cross(up).normalize_or_zero();
            let u2 = side.cross(t);
            let mut ring = Vec::new();
            for i in 0..segs {
                let a = std::f32::consts::TAU * i as f32 / segs as f32;
                let (c, s) = (a.cos(), a.sin());
                // Superellipse (rounded rectangle): |c|^0.5 sign.
                let sc = c.signum() * c.abs().powf(0.6);
                let ss = s.signum() * s.abs().powf(0.6);
                let q = p + side * (sc * hw) + u2 * (ss * hh);
                let n = side * (c / hw.max(1e-4)) + u2 * (s / hh.max(1e-4));
                ring.push(self.vert(q, n, mat, ao));
            }
            rings.push(ring);
        }
        self.strips(&rings, true);
        // Caps.
        for (k, ring) in [(0usize, rings[0].clone()), (path.len() - 1, rings[rings.len() - 1].clone())] {
            let t = (path[(k + 1).min(path.len() - 1)] - path[k.saturating_sub(1)]).normalize_or_zero();
            let s = if k == 0 { -1.0 } else { 1.0 };
            let c = self.vert(path[k], t * s, mat, ao);
            for i in 0..ring.len() {
                self.tri(c, ring[i], ring[(i + 1) % ring.len()]);
            }
        }
    }
}

/// A mesh and how it binds.
#[derive(Clone, Debug)]
pub struct Piece {
    pub name: String,
    pub mesh: Mesh,
    pub bind: Bind,
}

/// Polynomial smooth maximum (rounded intersection of two regions), k the blend
/// width (m).
pub fn smax(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.max(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.max(b) + h * h * k * 0.25
}

/// Polynomial smooth minimum.
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    -smax(-a, -b, k)
}

pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Cubic Hermite on [0, 1]: values y0, y1 and slopes m0, m1 (per unit t).
pub fn hermite(t: f32, y0: f32, y1: f32, m0: f32, m1: f32) -> f32 {
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * m1
}

/// A smooth closed curve through 2-D points (centripetal Catmull–Rom), sampled
/// about every `step` m. Used for every cut line and outline, so they are never
/// polygons at the scale anything can see.
pub fn smooth_closed(points: &[[f32; 2]], step: f32) -> Vec<[f32; 2]> {
    let n = points.len();
    let mut out = Vec::new();
    for i in 0..n {
        let p0 = glam::Vec2::from(points[(i + n - 1) % n]);
        let p1 = glam::Vec2::from(points[i]);
        let p2 = glam::Vec2::from(points[(i + 1) % n]);
        let p3 = glam::Vec2::from(points[(i + 2) % n]);
        let k = ((p2 - p1).length() / step).ceil().max(1.0) as usize;
        for s in 0..k {
            out.push(catmull(p0, p1, p2, p3, s as f32 / k as f32).to_array());
        }
    }
    out
}

fn catmull(p0: glam::Vec2, p1: glam::Vec2, p2: glam::Vec2, p3: glam::Vec2, t: f32) -> glam::Vec2 {
    let kt = |a: glam::Vec2, b: glam::Vec2| (a - b).length().max(1e-5).sqrt();
    let t0 = 0.0;
    let t1 = t0 + kt(p1, p0);
    let t2 = t1 + kt(p2, p1);
    let t3 = t2 + kt(p3, p2);
    let tt = t1 + (t2 - t1) * t;
    let a1 = p0 * ((t1 - tt) / (t1 - t0)) + p1 * ((tt - t0) / (t1 - t0));
    let a2 = p1 * ((t2 - tt) / (t2 - t1)) + p2 * ((tt - t1) / (t2 - t1));
    let a3 = p2 * ((t3 - tt) / (t3 - t2)) + p3 * ((tt - t2) / (t3 - t2));
    let b1 = a1 * ((t2 - tt) / (t2 - t0)) + a2 * ((tt - t0) / (t2 - t0));
    let b2 = a2 * ((t3 - tt) / (t3 - t1)) + a3 * ((tt - t1) / (t3 - t1));
    b1 * ((t2 - tt) / (t2 - t1)) + b2 * ((tt - t1) / (t2 - t1))
}

/// A closed 2-D curve (dense polyline) with a signed distance (negative inside).
#[derive(Clone, Debug)]
pub struct Shape {
    pub pts: Vec<[f32; 2]>,
    lo: [f32; 2],
    hi: [f32; 2],
}

impl Shape {
    /// A smooth outline through control points (Catmull–Rom, 5 mm samples).
    pub fn smooth(points: &[(f32, f32)]) -> Self {
        let p: Vec<[f32; 2]> = points.iter().map(|&(a, b)| [a, b]).collect();
        Self::from_pts(smooth_closed(&p, 0.005))
    }

    pub fn from_pts(pts: Vec<[f32; 2]>) -> Self {
        let mut lo = [f32::MAX; 2];
        let mut hi = [f32::MIN; 2];
        for p in &pts {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Self { pts, lo, hi }
    }

    /// Signed distance (negative inside); far outside the bounds, the distance
    /// to the bounding box (a lower bound, sign right).
    pub fn sd(&self, x: f32, y: f32) -> f32 {
        let dx = (self.lo[0] - x).max(x - self.hi[0]);
        let dy = (self.lo[1] - y).max(y - self.hi[1]);
        let out = dx.max(0.0).hypot(dy.max(0.0));
        if out > 0.08 {
            return out;
        }
        let p = &self.pts;
        let mut best = f32::MAX;
        let mut inside = false;
        let mut j = p.len() - 1;
        for i in 0..p.len() {
            let (a, b) = (p[j], p[i]);
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let l2 = ex * ex + ey * ey;
            let t = if l2 > 0.0 {
                (((x - a[0]) * ex + (y - a[1]) * ey) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (qx, qy) = (a[0] + t * ex - x, a[1] + t * ey - y);
            best = best.min(qx * qx + qy * qy);
            if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
                inside = !inside;
            }
            j = i;
        }
        let d = best.sqrt();
        if inside {
            -d
        } else {
            d
        }
    }
}
