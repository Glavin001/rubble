//! The body surface: one connected, dense triangle mesh that the panels are cut
//! from, built as surface patches between explicit feature curves.
//!
//! **Feature curves** (each a smooth function of the place round the car or
//! along it; the package curves are blurred first, so none has curvature
//! jumps at its control points):
//!
//! * the *belt* `L0`: the band's top edge round the whole plan outline;
//! * the *shoulder crease* and the *lower-door crease*: band rows;
//! * the *top line* `L1`: the belt offset inward by the ramp (the roof rail,
//!   the A- and C-pillars' upper edges, the bonnet's edges and leading edge,
//!   the deck's edges and trailing edge);
//! * the *deck ledge* `K`: the deck's side edge carried forward along the
//!   quarter's top, falling to the belt under the quarter light (the
//!   C-pillar's base);
//! * the bonnet's *power-dome* creases, and in the front fascia the grille's
//!   collar, the chin and the bumper's corner crease (arc-view curves).
//!
//! **Patches** between them:
//!
//! * **Band** (sides, bumpers, fascia, tail): a section swept round the plan
//!   outline by arc length θ, a cubic Hermite span between each pair of
//!   feature rows (curl → lower line → lower-door crease → shoulder crease →
//!   belt). The section blends between side, nose and tail shapes, so bumpers
//!   wrap round the corners without end caps. Arch flares, the lower-door
//!   step and the fascia's sculpture are offsets along the section.
//! * **Ramp** (fender tops, greenhouse sides and pillars, C-pillar sails,
//!   quarter tops, the bonnet's and deck's roll-overs): between `L0` and
//!   `L1`, per place round the car a section in (inward distance, height)
//!   whose slope steps between plateaus (`Ramp`): it leaves the belt on the
//!   band's own tangent (tangent continuous), rounds over a controlled
//!   width into a straight span, and rounds into the top on the top's own
//!   tangent at `L1`. Behind the cabin a second family runs the same way
//!   with a crisp roll onto the deck ledge and a fillet at `K` into the
//!   C-pillar's sail. Every step of slope is a smoothstep, so curvature is
//!   continuous everywhere, and a crisp line is a short window of known width.
//! * **Top** (bonnet, windscreen, roof, backlight, deck): an analytic crowned
//!   height field over plan view inside `L1`, with the power domes and the
//!   boot lip.
//!
//! **Meshing.** Each patch is meshed along its feature curves: band rows lie
//! on the crease and feature lines; the ramp's rows lie on its window edges
//! (the shoulder round, the ledge, the fillet, the rail round) and are spaced
//! by arc length between them, so they follow `L0`, `K` and `L1` round the car;
//! the top's rows are anchored on `L1` and a column lies on each dome crease.
//! Every normal is the surface's own (central differences of the patch
//! functions, not of the mesh), so highlights follow the surface, not the
//! triangles. Band columns are finer across the nose and the tail.
//!
//! Coordinates: definition coordinates (x across, +x the car's left; y up; z
//! from the rear end, 0, to the front, `length`).

use super::{hermite, smax, smoothstep, Mesh, Shape};
use crate::authoring::body::BodyProfile;
use crate::authoring::curve::Curve;
use glam::{Vec2, Vec3};

/// The band's section shape at one place round the car.
#[derive(Clone, Copy, Debug, Default)]
pub struct SectionShape {
    /// Lower line height above the band's bottom (m).
    pub lower: f32,
    /// Inward offset at the lower line (m, negative = inward).
    pub d_lower: f32,
    /// Crease height (absolute, m), or NaN to use the side's crease curve.
    pub crease: f32,
    /// Slopes (offset per height) just below and just above the crease: the
    /// crease's sharpness. Below positive (the body grows to the crease), above
    /// positive leans in.
    pub a_below: f32,
    pub a_above: f32,
    /// Offset at the top edge and its slope there (lean toward the lid).
    pub d_top: f32,
    pub m_top: f32,
    /// How far the bottom curls under (m).
    pub tuck: f32,
    /// The lower body's bow: its slope at the lower line as a multiple of the
    /// mean slope up to the crease (0: upright there). Above 1 the side is
    /// convex from the sill to the crease (it tucks under progressively, so
    /// the flank turns away from the light instead of reading as a slab).
    pub bow: f32,
}

impl SectionShape {
    fn mix(a: &Self, b: &Self, t: f32) -> Self {
        let l = |x: f32, y: f32| x + (y - x) * t;
        Self {
            lower: l(a.lower, b.lower),
            d_lower: l(a.d_lower, b.d_lower),
            crease: l(a.crease, b.crease),
            a_below: l(a.a_below, b.a_below),
            a_above: l(a.a_above, b.a_above),
            d_top: l(a.d_top, b.d_top),
            m_top: l(a.m_top, b.m_top),
            tuck: l(a.tuck, b.tuck),
            bow: l(a.bow, b.bow),
        }
    }
}

/// The body's styling (the surface's shape beyond the package curves).
#[derive(Clone, Debug)]
pub struct BodyStyle {
    /// Crease height along the sides (z → y).
    pub crease: Curve,
    pub side: SectionShape,
    pub nose: SectionShape,
    pub tail: SectionShape,
    /// Plan view: how far the nose and tail bow (m at the outline's corner) and
    /// the corner blend widths (m).
    pub nose_bow: f32,
    pub tail_bow: f32,
    pub nose_corner: f32,
    pub tail_corner: f32,
    /// Arch flare: outward swell (m) round each well, over this width beyond
    /// the arch radius.
    pub flare: f32,
    pub flare_width: f32,
    /// The rear arches' flare (m): fuller haunches over the driven wheels.
    pub rear_flare: f32,
    /// Roof crown outside the cabin (bonnet and deck), m.
    pub lid_crown: f32,
    /// Grid spacing (m) across the sections (band rows, ramp rows, top
    /// columns) and along the car (band columns): the body curves far more
    /// across than along, and every cut line is exact whatever the grid.
    pub step: f32,
    pub step_along: f32,
    /// The ramp's plan length (m) at the nose and at the tail (the sides run
    /// from the belt to the top line): short gives a crisp bonnet leading
    /// edge and a crisp boot deck.
    pub nose_ramp: f32,
    pub tail_ramp: f32,
    /// The rear deck ledge and the C-pillar: the deck's level carried forward
    /// from z = `rear_deck[0]`, falling to the belt by z = `rear_deck[1]`,
    /// the ledge's slope (inward), and the width (m) of the fillet where the
    /// C-pillar's sail rises off it. Zero width: no ledge.
    pub rear_deck: [f32; 4],
    /// Bonnet power dome: the creases' half spacing at the nose and at the
    /// cowl, the dome's rise between them and the creases' rounding (m); the
    /// bonnet's z span (cowl, nose).
    pub hood_crease: [f32; 4],
    pub hood_z: [f32; 2],
    /// Boot lip: the deck's rise (m) at its trailing edge, over this length
    /// (m) forward of it, within this half width.
    pub deck_lip: [f32; 3],
    /// Lower-door sculpture: a feature line in side view (z → y) below which
    /// the side steps in by `scoop_depth` (a crisp edge at the line, easing
    /// out toward the sill), faded in over z ∈ [scoop_z[0], scoop_z[1]] and
    /// out over [scoop_z[2], scoop_z[3]].
    pub scoop: Curve,
    pub scoop_depth: f32,
    pub scoop_z: [f32; 4],
    /// The ramp's shoulder round as a share of its length: on the cabin's
    /// sides (the glass leaves the belt almost at once) and over the front
    /// fenders (a soft shoulder into the fender top); the rail round's share
    /// at the bonnet's edges (smaller is crisper).
    pub shoulder: [f32; 2],
    pub bonnet_edge: f32,
    /// Front fascia sculpture, in arc view (a from the front centre, h from
    /// the band's top). The grille's outline and its collar: the bumper
    /// rises `collar[0]` m to a crest `collar[1]` m outside the opening,
    /// easing out over `collar[2]` m.
    pub grille: Vec<(f32, f32)>,
    pub collar: [f32; 3],
    /// The lower fascia's lips: the lower feature row sits at world height
    /// `lip[0]` across the nose and `lip[2]` across the tail (a crisp
    /// horizontal crease on a row), and below it the bumper stands
    /// `lip[1]` (the chin) and `lip[3]` m proud.
    pub lip: [f32; 4],
}

/// The old lid's ramp profile at t ∈ [0, 1] (0 at the edge, 1 at the top
/// line): straight for t ≤ 1 − tau, then a quartic round into the top with
/// zero curvature at both ends.
pub fn rail(t: f32, tau: f32) -> f32 {
    let m = 1.0 / (1.0 - 0.5 * tau);
    let t0 = 1.0 - tau;
    if t <= t0 {
        return m * t;
    }
    let s = t - t0;
    m * t0 + m * s - m / (tau * tau) * s * s * s + m / (2.0 * tau * tau * tau) * s * s * s * s
}

/// Quintic smoothstep: value, slope and curvature continuous at both ends.
pub fn smootherstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// The deck ledge's height at z over a belt at y_t: the deck's level
/// carried forward from zd0, falling (C2) to just under the belt by zd1.
pub fn ledge_y(prof: &Pkg, zd0: f32, zd1: f32, z: f32, y_t: f32) -> f32 {
    let deck0 = prof.top.eval(z.min(zd0));
    deck0 + (y_t - 0.01 - deck0) * smootherstep(zd0, zd1, z)
}

/// Smooth max(s, 0) with a rounding of w (m).
fn soft_pos(s: f32, w: f32) -> f32 {
    0.5 * (s + (s * s + w * w).sqrt())
}

/// The rise of a unit step of slope spread (smoothstep) over [c, c + a], at s.
fn jstep(s: f32, c: f32, a: f32) -> f32 {
    if a <= 1e-6 {
        return (s - c).max(0.0);
    }
    let u = (s - c) / a;
    if u <= 0.0 {
        0.0
    } else if u >= 1.0 {
        a * (u - 0.5)
    } else {
        a * u * u * u * (1.0 - 0.5 * u)
    }
}

/// A ramp section between the belt and the top line: height over inward
/// plan distance s ∈ [0, d] as plateaus of slope joined by smoothstep
/// windows: k0 (the band's slope at the belt) → ka over [0, a0] (the
/// shoulder round, or the crisp roll onto the deck ledge), ka → kb over
/// [sk − w, sk + w] (the C-pillar's fillet), kb → k1 (the top's slope at the
/// top line) over [d − a1, d] (the rail round). Slope is C1 and the end
/// windows also take the neighbours' curvature (κ0 the band's at the belt,
/// κ1 the top's at the top line), so curvature is continuous across both
/// feature curves and zero at the inner windows' ends.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ramp {
    pub d: f32,
    pub k0: f32,
    pub c0: f32,
    pub a0: f32,
    pub ka: f32,
    pub sk: f32,
    pub w: f32,
    pub kb: f32,
    pub a1: f32,
    pub k1: f32,
    pub c1: f32,
}

/// ∫ of the Hermite start-slope basis (u³ − 2u² + u) over [0, u], u ≤ 1.
fn q0(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    u * u * (0.5 + u * (-2.0 / 3.0 + 0.25 * u))
}

/// ∫ of the Hermite end-slope basis (u³ − u²) over [0, u], u ≤ 1.
fn q1(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    u * u * u * (0.25 * u - 1.0 / 3.0)
}

impl Ramp {
    pub fn h(&self, s: f32) -> f32 {
        let s = s.clamp(0.0, self.d);
        let mut y = self.k0 * s
            + (self.ka - self.k0) * jstep(s, 0.0, self.a0)
            + (self.kb - self.ka) * jstep(s, self.sk - self.w, 2.0 * self.w)
            + (self.k1 - self.kb) * jstep(s, self.d - self.a1, self.a1);
        if self.a0 > 1e-6 {
            y += self.c0 * self.a0 * self.a0 * q0(s / self.a0);
        }
        if self.a1 > 1e-6 {
            y += self.c1 * self.a1 * self.a1 * q1((s - (self.d - self.a1)) / self.a1);
        }
        y
    }

    /// Window edges: the mesh's rows lie on these.
    pub fn knots(&self) -> [f32; 7] {
        [
            0.0,
            self.a0,
            self.sk - self.w,
            self.sk,
            self.sk + self.w,
            self.d - self.a1,
            self.d,
        ]
    }

    /// One straight span (the glass, the fender top) between the shoulder
    /// round and the rail round, rising `hgt` over `d`.
    #[allow(clippy::too_many_arguments)]
    pub fn simple(d: f32, hgt: f32, k0: f32, c0: f32, k1: f32, c1: f32, a0: f32, a1: f32) -> Self {
        let ks = (hgt - k0 * a0 * 0.5 - k1 * a1 * 0.5 - c0 * a0 * a0 / 12.0 + c1 * a1 * a1 / 12.0)
            / (d - 0.5 * (a0 + a1)).max(1e-4);
        let span = (d - a1 - a0).max(0.0);
        Ramp {
            d,
            k0,
            c0,
            a0,
            ka: ks,
            sk: a0 + 0.5 * span,
            w: (0.25 * span).min(0.02),
            kb: ks,
            a1,
            k1,
            c1,
        }
    }

    /// The deck ledge: a crisp roll off the belt onto a ledge of slope `kq`
    /// at height `hk` (the roll's width solved so it lands there), a fillet
    /// at `sk` into the sail, the rail round into the top.
    #[allow(clippy::too_many_arguments)]
    pub fn ledge(
        d: f32,
        hgt: f32,
        k0: f32,
        c0: f32,
        k1: f32,
        c1: f32,
        kq: f32,
        hk: f32,
        sk: f32,
        w: f32,
        a1: f32,
        a0_min: f32,
    ) -> Self {
        let f = 0.1875 * w;
        let end = -k1 * a1 * 0.5 + c1 * a1 * a1 / 12.0;
        let kc = (hgt - hk - kq * f + end) / (d - sk - f - a1 * 0.5).max(1e-4);
        // The roll's rise: a0 (k0 − kq) / 2 + c0 a0² / 12.
        let rise = hk - kq * sk - f * (kc - kq);
        let (qa, qb) = (c0 / 12.0, 0.5 * (k0 - kq).max(0.3));
        let a0 = if qa.abs() < 1e-6 {
            rise / qb
        } else {
            let disc = (qb * qb + 4.0 * qa * rise).max(0.0);
            (-qb + disc.sqrt()) / (2.0 * qa)
        };
        let a0 = a0.clamp(a0_min, (sk - w - 0.002).max(a0_min));
        // With the roll fixed, the sail's slope lands the top line exactly.
        let rise = 0.5 * a0 * (k0 - kq) + c0 * a0 * a0 / 12.0;
        let kc = (hgt - rise - kq * sk + end) / ((d - sk) - a1 * 0.5).max(1e-4);
        Ramp {
            d,
            k0,
            c0,
            a0,
            ka: kq,
            sk,
            w,
            kb: kc,
            a1,
            k1,
            c1,
        }
    }
}

impl BodyStyle {
    /// The lower-door step-in (m, ≥ 0) at side-view (z, y), above the lower
    /// line yl, under the feature line ys.
    pub fn scoop_at(&self, z: f32, y: f32, yl: f32, ys: f32) -> f32 {
        if self.scoop_depth <= 0.0 {
            return 0.0;
        }
        let [a, b, c, d] = self.scoop_z;
        let fz = smoothstep(a, b, z) * (1.0 - smoothstep(c, d, z));
        if fz <= 0.0 {
            return 0.0;
        }
        // A crisp upper edge (1.2 cm), the full step below it, easing back out
        // toward the lower line.
        let edge = smoothstep(ys + 0.004, ys - 0.012, y);
        let ease = 0.35 + 0.65 * smoothstep(yl, ys - 0.02, y);
        self.scoop_depth * fz * edge * ease
    }

    /// Detail heights (m) on the top at plan (x, z): the bonnet's power
    /// dome and the boot lip. `z_tail` is the tail edge's z at this x.
    pub fn lid_detail(&self, x: f32, z: f32, z_tail: f32) -> f32 {
        let mut d = 0.0;
        let [xn, xc, rise, w] = self.hood_crease;
        let [zc, zn] = self.hood_z;
        if rise > 0.0 && z > zc - 0.05 {
            let t = ((z - zc) / (zn - zc)).clamp(0.0, 1.0);
            let xl = xc + (xn - xc) * t;
            let fade =
                smoothstep(zc - 0.03, zc + 0.10, z) * (1.0 - smoothstep(zn - 0.22, zn - 0.04, z));
            // Flat between the creases, a crisp break at them, falling away
            // outside.
            let s = soft_pos(x.abs() - xl, w);
            d += rise * fade * (1.0 - smoothstep(0.0, 0.17, s));
        }
        let [lip, len, hw] = self.deck_lip;
        if lip > 0.0 && z < z_tail + len {
            let along = smoothstep(z_tail + len, z_tail + 0.015, z);
            let across = 1.0 - smoothstep(hw - 0.12, hw, x.abs());
            d += lip * along * along * across;
        }
        d
    }

    /// The power domes' crease half spacing at z (continued at the cowl's
    /// spacing behind the bonnet).
    pub fn crease_x(&self, z: f32) -> f32 {
        let [xn, xc, _, _] = self.hood_crease;
        let [zc, zn] = self.hood_z;
        let t = ((z - zc) / (zn - zc)).clamp(0.0, 1.0);
        xc + (xn - xc) * t
    }
}

/// One row of the lid's frame: at z, the edge (x_t ≥ 0, y_t), the top line's
/// half width x_r and its height y_e.
#[derive(Clone, Copy, Debug)]
pub struct LidRow {
    pub z: f32,
    pub x_t: f32,
    pub y_t: f32,
    pub x_r: f32,
    pub y_e: f32,
}

pub struct BodySurface {
    pub mesh: Mesh,
    /// Per vertex: arc distance (m) round the plan outline from the front
    /// centre (the ramp's and top's: their belt column's), and whether it is
    /// above the belt (ramp and top: the old "lid").
    pub arc: Vec<f32>,
    pub lid: Vec<bool>,
    /// Per vertex: the band's top-edge height at its place round the car (the
    /// arc view's reference height).
    pub ytop: Vec<f32>,
    /// Per vertex: grid (column, row) of band vertices (u32::MAX above the belt).
    pub grid: Vec<[u32; 2]>,
    /// Per vertex: the patch it lies on (0 band, 1 ramp, 2 top).
    pub patch: Vec<u8>,
    pub band: Band,
    pub rows: Vec<LidRow>,
    /// The band's top-edge vertices round the loop (column order).
    pub top_edge: Vec<u32>,
    /// The band's grid: columns round the loop, rows bottom to top.
    pub cols: usize,
    pub band_rows: usize,
}

/// The band: plan outline and sections.
#[derive(Clone)]
pub struct Band {
    pub prof: Pkg,
    pub style: BodyStyle,
    /// The style's crease line, blurred like the package curves.
    crease: Lut,
    /// Plan outline (x ≥ 0 half), uniformly by arc length: (point (x, z), outward normal).
    outline: Vec<(Vec2, Vec2)>,
    /// Half outline length (rear centre to front centre).
    pub half: f32,
    /// Row fractions of the crease and lower line.
    pub v_lower: f32,
    pub v_crease: f32,
    /// Row fraction of the lower-door feature line (a row lies on it, so its
    /// crisp edge runs straight along the grid).
    pub v_scoop: f32,
    /// The fascia's arc-view outlines (grille collar, corner crease).
    grille: Option<Shape>,
}

#[derive(Clone, Copy, Debug)]
pub struct Section {
    pub theta: f32,
    pub p: Vec2,
    pub n: Vec2,
    pub yb: f32,
    pub yl: f32,
    pub yc: f32,
    /// The lower-door feature line's height here (kept between the lower line
    /// and the crease).
    pub ys: f32,
    pub yt: f32,
    pub s: SectionShape,
    /// Nose and tail weights.
    pub wf: f32,
    pub wr: f32,
    /// 1 where the top line barely rises over the belt (fender tops, deck):
    /// the shoulder rolls over toward the top there.
    pub roll: f32,
}

impl Band {
    pub fn new(prof: &Pkg, style: &BodyStyle) -> Self {
        let l = prof.length;
        let w = |z: f32| prof.half_width.eval(z) + prof.side_bulge;
        let xs = 0.9f32;
        let f = |x: f32, z: f32| -> f32 {
            let a = x - w(z);
            let b = z - (l - style.nose_bow * (x / xs).powi(2));
            let c = style.tail_bow * (x / xs).powi(2) - z;
            smax(smax(a, b, style.nose_corner), c, style.tail_corner)
        };
        // Rays from the plan centre (the outline is convex).
        let c = Vec2::new(0.0, 0.5 * l);
        let nray = 4000;
        let mut pts = Vec::with_capacity(nray + 1);
        for i in 0..=nray {
            let a = std::f32::consts::PI * i as f32 / nray as f32;
            let d = Vec2::new(a.sin(), -a.cos());
            let (mut lo, mut hi) = (0.0f32, 4.0f32);
            for _ in 0..40 {
                let m = 0.5 * (lo + hi);
                let q = c + d * m;
                if f(q.x, q.y) < 0.0 {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            let q = c + d * lo;
            pts.push(Vec2::new(if i == 0 || i == nray { 0.0 } else { q.x }, q.y));
        }
        // Resample by arc length (5 mm).
        let mut acc = vec![0.0f32];
        for w in pts.windows(2) {
            acc.push(acc.last().unwrap() + w[0].distance(w[1]));
        }
        let half = *acc.last().unwrap();
        let n = (half / 0.005).ceil() as usize;
        let mut outline = Vec::with_capacity(n + 1);
        let mut j = 0;
        let at = |s: f32, j: &mut usize| -> Vec2 {
            while *j + 2 < acc.len() && acc[*j + 1] < s {
                *j += 1;
            }
            let t = ((s - acc[*j]) / (acc[*j + 1] - acc[*j]).max(1e-9)).clamp(0.0, 1.0);
            pts[*j].lerp(pts[*j + 1], t)
        };
        let samples: Vec<Vec2> = (0..=n)
            .map(|i| at(half * i as f32 / n as f32, &mut j))
            .collect();
        for i in 0..=n {
            let a = if i == 0 {
                Vec2::new(-samples[1].x, samples[1].y)
            } else {
                samples[i - 1]
            };
            let b = if i == n {
                Vec2::new(-samples[n - 1].x, samples[n - 1].y)
            } else {
                samples[i + 1]
            };
            let t = (b - a).normalize_or_zero();
            let mut nn = Vec2::new(t.y, -t.x);
            if nn.dot(samples[i] - c) < 0.0 {
                nn = -nn;
            }
            outline.push((samples[i], nn));
        }
        Self {
            prof: prof.clone(),
            style: style.clone(),
            crease: blur_curve(&style.crease, l, 0.12),
            outline,
            half,
            v_lower: 0.0,
            v_crease: 0.0,
            v_scoop: 0.0,
            grille: if style.grille.len() >= 3 && style.collar[0] > 0.0 {
                Some(Shape::smooth(&style.grille))
            } else {
                None
            },
        }
    }

    /// Plan point and outward normal at θ ∈ [0, half] (x ≥ 0 side).
    pub fn plan(&self, theta: f32) -> (Vec2, Vec2) {
        let o = &self.outline;
        let n = o.len() - 1;
        let u = (theta / self.half).clamp(0.0, 1.0) * n as f32;
        let i = (u.floor() as usize).min(n - 1);
        let t = u - i as f32;
        // Past the centre line: the mirror image (x → −x).
        let at = |k: isize| -> (Vec2, Vec2) {
            if k < 0 {
                let (p, nn) = o[(-k) as usize];
                (Vec2::new(-p.x, p.y), Vec2::new(-nn.x, nn.y))
            } else if k as usize > n {
                let (p, nn) = o[2 * n - k as usize];
                (Vec2::new(-p.x, p.y), Vec2::new(-nn.x, nn.y))
            } else {
                o[k as usize]
            }
        };
        let k = i as isize;
        let (a, b, c, d) = (at(k - 1), at(k), at(k + 1), at(k + 2));
        (
            cubic(a.0, b.0, c.0, d.0, t),
            cubic(a.1, b.1, c.1, d.1, t).normalize_or_zero(),
        )
    }

    /// Columns round the half loop (θ from the rear centre to the front
    /// centre): `step` apart along the sides, `1 − fine` of that across the
    /// nose and the tail (where the fascia's sculpture is).
    pub fn column_thetas(&self, step: f32, fine: f32) -> Vec<f32> {
        let n = 2000;
        let dt = self.half / n as f32;
        let mut u = vec![0.0f32; n + 1];
        for i in 1..=n {
            let (_, nn) = self.plan(dt * (i as f32 - 0.5));
            let w = smoothstep(0.25, 0.97, nn.y).max(smoothstep(0.25, 0.97, -nn.y));
            u[i] = u[i - 1] + dt / (step * (1.0 - fine * w));
        }
        let h = (u[n].ceil() as usize).max(4);
        let mut out = Vec::with_capacity(h + 1);
        let mut i = 0;
        for c in 0..=h {
            let target = u[n] * c as f32 / h as f32;
            while i + 1 < n && u[i + 1] < target {
                i += 1;
            }
            let t = ((target - u[i]) / (u[i + 1] - u[i]).max(1e-9)).clamp(0.0, 1.0);
            out.push(if c == h {
                self.half
            } else {
                dt * (i as f32 + t)
            });
        }
        out
    }

    /// The section at θ.
    pub fn section(&self, theta: f32) -> Section {
        let (p, n) = self.plan(theta);
        let st = &self.style;
        let wf = smoothstep(0.25, 0.97, n.y);
        let wr = smoothstep(0.25, 0.97, -n.y);
        let mut side = st.side;
        side.crease = self.crease.eval(p.y);
        // Fenders and deck sides roll toward the flat top; the cabin's doors
        // lean in less (the glass continues them).
        // The rise averaged over ±30 cm along the car: at the cowl the top
        // line climbs the windscreen within 20 cm, and a roll that followed
        // it point by point would twist the shoulder there.
        let rise = [-0.3f32, -0.15, 0.0, 0.15, 0.3]
            .iter()
            .map(|dz| {
                let z = (p.y + dz).clamp(0.0, self.prof.length);
                (self.prof.top.eval(z) - self.prof.shoulder.eval(z)).max(0.0)
            })
            .sum::<f32>()
            / 5.0;
        let roll = 1.0 - smootherstep(0.08, 0.30, rise);
        // Not behind the cabin when the deck ledge is on: there the quarter
        // stays upright into the ledge's crisp roll (a hard shoulder).
        let front = if st.rear_deck[3] > 0.0 {
            smoothstep(st.rear_deck[1] - 0.25, st.rear_deck[1] + 0.35, p.y)
        } else {
            1.0
        };
        side.m_top += 1.1 * roll * front;
        let mut s = side;
        if wf > 0.0 {
            s = SectionShape::mix(&s, &st.nose, wf);
        }
        if wr > 0.0 {
            s = SectionShape::mix(&s, &st.tail, wr);
        }
        let yb = self.prof.bottom.eval(p.y);
        let yt = self.prof.shoulder.eval(p.y);
        let yl = (yb + s.lower).min(yt - 0.1);
        let yc = s.crease.clamp(yl + 0.04, yt - 0.04);
        // The lower feature row: the lower-door crease on the sides, the
        // lips' crease across the nose and the tail.
        let side_ys = st.scoop.eval(p.y);
        let ys = side_ys + (st.lip[0] - side_ys) * wf + (st.lip[2] - side_ys) * wr;
        let ys = ys.clamp(yl + 0.15 * (yc - yl), yl + 0.75 * (yc - yl));
        Section {
            theta,
            p,
            n,
            yb,
            yl,
            yc,
            ys,
            yt,
            s,
            wf,
            wr,
            roll: roll * front,
        }
    }

    /// (offset, height) of the section at row fraction v.
    pub fn section_point(&self, sc: &Section, v: f32) -> Vec2 {
        let s = &sc.s;
        let (vl, vc) = (self.v_lower, self.v_crease);
        if v <= vl {
            let phi = (v / vl).clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
            let c = Vec2::new(s.d_lower - s.tuck, sc.yl);
            return c + Vec2::new(s.tuck * phi.sin(), -(sc.yl - sc.yb) * phi.cos());
        }
        if v <= vc {
            // Piecewise in v so that a row lies on the feature line ys.
            let dy = sc.yc - sc.yl;
            let vs = self.v_scoop.clamp(vl + 1e-4, vc - 1e-4);
            let ts = ((sc.ys - sc.yl) / dy).clamp(0.05, 0.95);
            let t = if v <= vs {
                ts * (v - vl) / (vs - vl)
            } else {
                ts + (1.0 - ts) * (v - vs) / (vc - vs)
            };
            let y = sc.yl + dy * t;
            let d = hermite(t, s.d_lower, 0.0, -s.d_lower * s.bow, s.a_below * dy);
            return Vec2::new(d, y);
        }
        let t = (v - vc) / (1.0 - vc);
        let dy = sc.yt - sc.yc;
        let y = sc.yc + dy * t;
        let d = hermite(t, 0.0, s.d_top, -s.a_above * dy, -s.m_top * dy);
        Vec2::new(d, y)
    }

    /// The arch flare (outward offset) at plan point z and height y.
    fn flare(&self, z: f32, y: f32) -> f32 {
        let mut best = 0.0f32;
        for w in &self.prof.wells {
            let r = Vec2::new(z - w.z, y - w.y).length();
            let rear = w.z < 0.5 * self.prof.length;
            let (amt, width) = if rear {
                // The haunch over a tighter band: a firmer shoulder above
                // the rear wheel.
                (self.style.rear_flare, self.style.flare_width * 0.8)
            } else {
                (self.style.flare, self.style.flare_width)
            };
            // A swell that rises over the flare width to a plateau at the lip.
            let k = smoothstep(w.radius + width, w.radius + 0.03, r);
            best = best.max(amt * k);
        }
        best
    }

    /// The front fascia's sculpture (outward offset, m) at height y: the
    /// collar round the grille.
    fn fascia(&self, sc: &Section, y: f32) -> f32 {
        let nose = smoothstep(0.0, 0.35, sc.wf);
        if nose <= 0.0 {
            return 0.0;
        }
        let st = &self.style;
        let a = self.half - sc.theta;
        let h = y - sc.yt;
        let mut d = 0.0f32;
        if let Some(g) = &self.grille {
            let [ht, crest, width] = st.collar;
            let sd = g.sd(a, h);
            if sd < crest + width {
                let k = if sd >= crest {
                    let t = ((crest + width - sd) / width).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                } else {
                    // Over the crest and down toward the opening.
                    let t = ((crest - sd) / crest).clamp(0.0, 1.6);
                    1.0 - 0.55 * t * t
                };
                d += ht * k;
            }
        }
        d * nose
    }

    /// The lips (outward offset, m) at height y: below the lower feature
    /// row the nose's chin and the tail's bumper stand proud, a crisp edge
    /// on the row easing back toward the curl.
    fn lips(&self, sc: &Section, y: f32) -> f32 {
        let [_, nd, _, td] = self.style.lip;
        let amt = nd * smoothstep(0.2, 0.6, sc.wf) + td * smoothstep(0.2, 0.6, sc.wr);
        if amt <= 0.0 {
            return 0.0;
        }
        let edge = smoothstep(sc.ys + 0.003, sc.ys - 0.010, y);
        let ease = 0.55 + 0.45 * smoothstep(sc.yl - 0.02, sc.ys - 0.015, y);
        amt * edge * ease
    }

    /// The surface point at θ (x ≥ 0 side) and row fraction v.
    pub fn point(&self, theta: f32, v: f32) -> Vec3 {
        let sc = self.section(theta);
        self.point_in(&sc, v)
    }

    pub fn point_in(&self, sc: &Section, v: f32) -> Vec3 {
        let q = self.section_point(sc, v);
        let side_w = 1.0 - sc.wf.max(sc.wr);
        let sidish = smoothstep(-0.2, 0.6, sc.n.x);
        let d = q.x + side_w * self.flare(sc.p.y, q.y) * sidish
            - side_w * sidish * self.style.scoop_at(sc.p.y, q.y, sc.yl, sc.ys)
            + self.fascia(sc, q.y)
            + self.lips(sc, q.y);
        Vec3::new(sc.p.x + sc.n.x * d, q.y, sc.p.y + sc.n.y * d)
    }

    /// The surface normal at a section (and its neighbours θ ∓ ε, `s3`) and
    /// row fraction v: central differences of the band's own function (a
    /// crease row gets the mean of its two sides' normals).
    fn normal_in(&self, s3: &[Section; 3], v: f32) -> Vec3 {
        let dv = 0.0006;
        let t1 = self.point_in(&s3[2], v) - self.point_in(&s3[0], v);
        let t2 =
            self.point_in(&s3[1], (v + dv).min(1.0)) - self.point_in(&s3[1], (v - dv).max(0.0));
        let n = t1.cross(t2).normalize_or_zero();
        let p = self.point_in(&s3[1], v);
        let l = self.prof.length;
        let out = p - Vec3::new(0.0, 0.55, p.z.clamp(1.0, l - 1.0));
        if n.dot(out) < 0.0 {
            -n
        } else {
            n
        }
    }

    /// The row fraction at height y (above the lower line; the curl below by
    /// bisection).
    pub fn v_at(&self, sc: &Section, y: f32) -> f32 {
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..30 {
            let m = 0.5 * (lo + hi);
            if self.section_point(sc, m).y < y {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    }

    /// θ at an arc distance `a` from the front centre.
    pub fn theta_front(&self, a: f32) -> f32 {
        self.half - a
    }
}

/// A curve sampled every 1 cm (O(1) lookups, linear between samples).
#[derive(Clone, Debug)]
pub struct Lut {
    z0: f32,
    dz: f32,
    v: Vec<f32>,
}

/// Catmull–Rom through four samples at t ∈ [0, 1] between the middle two:
/// value and slope continuous (a surface's normals from central differences
/// would step at every sample of a linear interpolation).
fn cubic<T>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T
where
    T: Copy
        + std::ops::Mul<f32, Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>,
{
    let (t2, t3) = (t * t, t * t * t);
    p1 * (1.0 - 3.0 * t2 + 2.0 * t3)
        + p2 * (3.0 * t2 - 2.0 * t3)
        + (p2 - p0) * (0.5 * (t - 2.0 * t2 + t3))
        + (p3 - p1) * (0.5 * (t3 - t2))
}

impl Lut {
    pub fn eval(&self, z: f32) -> f32 {
        let n = self.v.len();
        let u = ((z - self.z0) / self.dz).clamp(0.0, (n - 1) as f32);
        let i = (u.floor() as usize).min(n - 2);
        let t = u - i as f32;
        let v = &self.v;
        // Linear extrapolation past the ends.
        let p0 = if i == 0 { 2.0 * v[0] - v[1] } else { v[i - 1] };
        let p3 = if i + 2 >= n {
            2.0 * v[n - 1] - v[n - 2]
        } else {
            v[i + 2]
        };
        cubic(p0, v[i], v[i + 1], p3, t)
    }
}

/// The body's package as the surface uses it: the profile's curves blurred
/// along the car and tabulated.
#[derive(Clone, Debug)]
pub struct Pkg {
    pub length: f32,
    pub top: Lut,
    pub bottom: Lut,
    pub shoulder: Lut,
    pub half_width: Lut,
    pub top_half_width: Lut,
    pub crown: f32,
    pub side_bulge: f32,
    pub top_radius: f32,
    pub wells: Vec<crate::authoring::body::Well>,
}

/// A curve over z ∈ [0, l] blurred along the car (Gaussian, σ m) and
/// tabulated every 1 cm: a monotone cubic's curvature jumps at its control
/// points, which a clearcoat's reflections show as ripples or kinks.
pub fn blur_curve(c: &Curve, l: f32, sigma: f32) -> Lut {
    let dz = 0.01f32;
    let n = (l / dz).ceil() as usize;
    let raw: Vec<f32> = (0..=n).map(|i| c.eval(l * i as f32 / n as f32)).collect();
    let r = (3.0 * sigma / dz).ceil() as i32;
    let g: Vec<f32> = (-r..=r)
        .map(|k| (-(k as f32 * dz).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect();
    let v = (0..=n as i32)
        .map(|i| {
            let (mut s, mut w) = (0.0f32, 0.0f32);
            for k in -r..=r {
                // Odd reflection at the ends (keeps the end values and slopes).
                let j = i + k;
                let v = if j < 0 {
                    2.0 * raw[0] - raw[(-j).min(n as i32) as usize]
                } else if j > n as i32 {
                    2.0 * raw[n] - raw[(2 * n as i32 - j).max(0) as usize]
                } else {
                    raw[j as usize]
                };
                let gk = g[(k + r) as usize];
                s += gk * v;
                w += gk;
            }
            s / w
        })
        .collect();
    Lut {
        z0: 0.0,
        dz: l / n as f32,
        v,
    }
}

pub fn smooth_profile(p: &BodyProfile, sigma: f32) -> Pkg {
    let l = p.length;
    let blur = |c: &Curve| -> Lut { blur_curve(c, l, sigma) };
    Pkg {
        length: l,
        top: blur(&p.top),
        bottom: blur(&p.bottom),
        shoulder: blur(&p.shoulder),
        half_width: blur(&p.half_width),
        top_half_width: blur(&p.top_half_width),
        crown: p.crown,
        side_bulge: p.side_bulge,
        top_radius: p.top_radius,
        wells: p.wells.clone(),
    }
}

/// The top surface (bonnet, windscreen, roof, backlight, deck) and the
/// ramp sections that join it to the belt.
struct Top<'a> {
    band: &'a Band,
    /// The belt's z at the tail for |x| (the boot lip's reference).
    tail: Vec<(f32, f32)>,
    /// The ramp's plan length by θ, blurred along the belt (σ 8 cm): it
    /// falls from the fenders' 20 cm to the nose's 4.5 cm round the
    /// corners, and a fast change there would twist the corner's surface.
    ramp_len: Lut,
}

/// A ramp column: its belt point and plan normal, the two ramp families
/// and the deck family's weight.
#[derive(Clone, Copy, Debug)]
struct Col {
    p0: Vec3,
    n: Vec2,
    ra: Ramp,
    rb: Ramp,
    wb: f32,
}

impl Col {
    fn d(&self) -> f32 {
        self.ra.d
    }

    fn h(&self, s: f32) -> f32 {
        let a = self.ra.h(s);
        if self.wb > 0.0 {
            a + (self.rb.h(s) - a) * self.wb
        } else {
            a
        }
    }

    fn at(&self, s: f32) -> Vec3 {
        let s = s.clamp(0.0, self.d());
        Vec3::new(
            self.p0.x - self.n.x * s,
            self.p0.y + self.h(s),
            self.p0.z - self.n.y * s,
        )
    }

    /// The rows' window edges (the two families' blended), at least 0.5 mm
    /// apart.
    fn knots(&self) -> [f32; 7] {
        let (a, b) = (self.ra.knots(), self.rb.knots());
        let mut k = [0.0f32; 7];
        for i in 0..7 {
            k[i] = a[i] + (b[i] - a[i]) * self.wb;
        }
        let d = self.d();
        k[0] = 0.0;
        let gap = (0.002f32).min(d / 14.0);
        for i in 1..7 {
            k[i] = k[i].max(k[i - 1] + gap);
        }
        k[6] = d;
        for i in (1..6).rev() {
            k[i] = k[i].min(k[i + 1] - gap);
        }
        k
    }
}

impl Top<'_> {
    fn new(band: &Band) -> Top<'_> {
        // The belt round the tail: (x, z) from the rear centre to the side.
        let mut tail = Vec::new();
        let n = 400;
        for i in 0..=n {
            let th = band.half * 0.5 * i as f32 / n as f32;
            let (_, nn) = band.plan(th);
            if nn.y > -0.15 && i > 0 {
                break;
            }
            let p = band.point(th, 1.0);
            if tail.last().is_some_and(|q: &(f32, f32)| p.x <= q.0) {
                continue;
            }
            tail.push((p.x, p.z));
        }
        // The ramp's length round the belt, blurred.
        let st = &band.style;
        let prof = &band.prof;
        let dz = 0.01f32;
        let n = (band.half / dz).ceil() as usize;
        let raw: Vec<f32> = (0..=n)
            .map(|i| {
                let th = band.half * i as f32 / n as f32;
                let (p, nn) = band.plan(th);
                let z = p.y;
                let wf = smoothstep(0.2, 0.95, nn.y);
                let wr = smoothstep(0.2, 0.95, -nn.y);
                let side = (prof.half_width.eval(z) + prof.side_bulge
                    - (prof.top_half_width.eval(z) - prof.top_radius))
                    .max(0.08);
                side + (st.nose_ramp - side) * wf + (st.tail_ramp - side) * wr
            })
            .collect();
        let sigma = 0.08f32;
        let r = (3.0 * sigma / dz).ceil() as i32;
        let v: Vec<f32> = (0..=n as i32)
            .map(|i| {
                let (mut s, mut w) = (0.0f32, 0.0f32);
                for k in -r..=r {
                    // Even reflection at the centre line (the loop is symmetric).
                    let j = (i + k).abs();
                    let j = if j > n as i32 { 2 * n as i32 - j } else { j } as usize;
                    let g = (-((k as f32 * dz).powi(2)) / (2.0 * sigma * sigma)).exp();
                    s += g * raw[j];
                    w += g;
                }
                s / w
            })
            .collect();
        let ramp_len = Lut {
            z0: 0.0,
            dz: band.half / n as f32,
            v,
        };
        Top {
            band,
            tail,
            ramp_len,
        }
    }

    fn tail_z(&self, x: f32) -> f32 {
        let x = x.abs();
        let t = &self.tail;
        let i = t.partition_point(|q| q.0 < x);
        if i == 0 {
            return t[0].1;
        }
        if i >= t.len() {
            return t[t.len() - 1].1;
        }
        let (a, b) = (t[i - 1], t[i]);
        a.1 + (b.1 - a.1) * ((x - a.0) / (b.0 - a.0).max(1e-6))
    }

    /// The top's height at plan (x, z): the top line with its crown, the
    /// power domes and the boot lip.
    fn y(&self, x: f32, z: f32) -> f32 {
        let prof = &self.band.prof;
        let st = &self.band.style;
        let x_r = (prof.top_half_width.eval(z) - prof.top_radius).max(0.05);
        let l = prof.length;
        let cabin = smoothstep(0.55, 0.95, z) * (1.0 - smoothstep(l - 1.35, l - 0.95, z));
        let crown = st.lid_crown + (prof.crown - st.lid_crown) * cabin;
        let u = (x.abs() / x_r).min(1.0);
        prof.top.eval(z) + crown * (1.0 - u * u).powi(3) + st.lid_detail(x, z, self.tail_z(x))
    }

    fn normal(&self, x: f32, z: f32) -> Vec3 {
        let e = 0.001;
        let dx = (self.y(x + e, z) - self.y(x - e, z)) / (2.0 * e);
        let dz = (self.y(x, z + e) - self.y(x, z - e)) / (2.0 * e);
        Vec3::new(-dx, 1.0, -dz).normalize()
    }

    /// The ramp column at θ.
    fn col(&self, theta: f32) -> Col {
        let band = self.band;
        let st = &band.style;
        let prof = &band.prof;
        let theta = theta.clamp(0.0, band.half);
        let sc = band.section(theta);
        let p0 = band.point_in(&sc, 1.0);
        let n = sc.n;
        // The band's slope and curvature at the belt (height over inward
        // distance): the ramp leaves on them (curvature continuous).
        let dv = 0.003;
        let pts = [
            band.point_in(&sc, 1.0 - 2.0 * dv),
            band.point_in(&sc, 1.0 - dv),
            p0,
        ];
        let si = |p: Vec3| -((p.x - p0.x) * n.x + (p.z - p0.z) * n.y);
        let (s0, s1, s2) = (si(pts[0]), si(pts[1]), 0.0);
        let (ka_, kb_) = (
            (pts[1].y - pts[0].y) / (s1 - s0),
            (pts[2].y - pts[1].y) / (s2 - s1),
        );
        let (k0, c0) = if s1 - s0 > 1e-5 && s2 - s1 > 1e-5 {
            let c = ((kb_ - ka_) / (0.5 * (s2 - s0))).clamp(-60.0, 60.0);
            let k = kb_ + c * 0.5 * (s2 - s1);
            if k > 8.0 || k < 0.2 {
                (k.clamp(0.2, 8.0), 0.0)
            } else {
                (k, c)
            }
        } else {
            (8.0, 0.0)
        };
        let z = p0.z;
        let wf = smoothstep(0.35, 0.92, n.y);
        let wr = smoothstep(0.35, 0.92, -n.y);
        let d = self.ramp_len.eval(theta);
        let q1 = Vec2::new(p0.x - n.x * d, p0.z - n.y * d);
        let y1 = self.y(q1.x, q1.y);
        let hgt = (y1 - p0.y).max(0.004);
        let e = 0.004;
        let (yi, yo) = (
            self.y(q1.x - n.x * e, q1.y - n.y * e),
            self.y(q1.x + n.x * e, q1.y + n.y * e),
        );
        let k1 = (yi - yo) / (2.0 * e);
        let c1 = ((yi - 2.0 * y1 + yo) / (e * e)).clamp(-30.0, 30.0);
        // The rounds: a short shoulder on the cabin's sides, a soft one over
        // the fenders; the rail round long and gentle where the top barely
        // rises (fender tops), short under the roof; crisper at the bonnet's
        // edges; short at the nose and the tail.
        let rise_row = (prof.top.eval(z) - p0.y).max(0.004);
        let tau_side = (0.7 - (rise_row - 0.05) * 1.6).clamp(0.22, 0.7);
        let fender = sc.roll * smoothstep(0.5 * prof.length, 0.5 * prof.length + 0.4, z);
        let tau = tau_side + (st.bonnet_edge - tau_side) * fender;
        let alpha = st.shoulder[0] + (st.shoulder[1] - st.shoulder[0]) * sc.roll;
        let ends = wf.max(wr);
        let alpha = alpha + (0.3 - alpha) * ends;
        let tau = tau + (0.45 - tau) * ends;
        let (mut a0, mut a1) = (alpha * d, tau * d);
        if a0 + a1 > 0.9 * d {
            let k = 0.9 * d / (a0 + a1);
            a0 *= k;
            a1 *= k;
        }
        let ra = Ramp::simple(d, hgt, k0, c0, k1, c1, a0, a1);
        // The deck ledge (behind the cabin, on the sides).
        let [zd0, zd1, kq, fw] = st.rear_deck;
        let wb = if fw > 0.0 {
            smoothstep(0.3, 0.7, n.x) * (1.0 - smootherstep(zd1 - 0.2, zd1, z))
        } else {
            0.0
        };
        let rb = if wb > 0.0 {
            let w = 0.5 * fw;
            let hk = (ledge_y(prof, zd0, zd1, z, p0.y) - p0.y).clamp(0.002, hgt);
            // As the ledge comes down to the belt it tilts up to the glass's
            // slope, so it merges into the shoulder without a notch.
            let psi = 1.0 - smoothstep(0.0, 0.035, hk);
            let kq = kq + (ra.ka - kq) * psi;
            let a0e = (2.0 * hk / (k0 - kq).max(0.5)).max(0.012);
            let lo = a0e + w + 0.003;
            let hi = d - a1 - w - 0.003;
            // The sail's base sweeps out from the top line at the backlight's
            // foot to the belt under the quarter light.
            let phi = smootherstep(zd0 - 0.05, zd1 - 0.1, z);
            let sk = if hi > lo {
                hi + (lo - hi) * phi
            } else {
                0.5 * (lo + hi)
            };
            Ramp::ledge(d, hgt, k0, c0, k1, c1, kq, hk, sk, w, a1, 0.012)
        } else {
            ra
        };
        Col { p0, n, ra, rb, wb }
    }
}

impl BodySurface {
    pub fn build(prof: &BodyProfile, style: &BodyStyle) -> Self {
        // The package curves are monotone cubics through a handful of points:
        // smooth in value and slope but with curvature jumps at the points,
        // which a clearcoat's reflections show as ripples. The surface uses
        // them blurred (Gaussian, σ = 8 cm along the car).
        let prof = &smooth_profile(prof, 0.08);
        let mut band = Band::new(prof, style);
        let step = style.step;
        // Rows: curl, lower body, upper body (the crease and lower line on rows).
        let (r_curl, r_low, r_up) = (
            (0.16 / step).ceil() as usize,
            (0.55 / step).ceil() as usize,
            (0.28 / step).ceil() as usize,
        );
        let nr0 = r_curl + r_low + r_up;
        band.v_lower = r_curl as f32 / nr0 as f32;
        band.v_crease = (r_curl + r_low) as f32 / nr0 as f32;
        band.v_scoop = (r_curl + (r_low as f32 * 0.45).round() as usize) as f32 / nr0 as f32;
        // Row fractions: uniform, plus a pair of rows a few millimetres either
        // side of the crease and of the lower-door feature line, so the
        // normals turn within a tight fillet there (a crisp line of light).
        let (vl, vc, vs) = (band.v_lower, band.v_crease, band.v_scoop);
        let mut vrows: Vec<f32> = (0..=nr0).map(|r| r as f32 / nr0 as f32).collect();
        for v in [
            vc - (vc - vs) * 0.018,
            vc + (1.0 - vc) * 0.03,
            vs - (vs - vl) * 0.035,
            vs + (vc - vs) * 0.03,
        ] {
            vrows.push(v);
        }
        vrows.sort_by(|a, b| a.total_cmp(b));
        vrows.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
        let nrows = vrows.len() - 1;
        let thetas = band.column_thetas(style.step_along, 0.3);
        let h = thetas.len() - 1;
        let ncol = 2 * h;
        let l = prof.length;
        let mut mesh = Mesh::default();
        let mut arc = Vec::new();
        let mut lid = Vec::new();
        let mut ytop: Vec<f32> = Vec::new();
        let mut grid = Vec::new();
        let mut patch: Vec<u8> = Vec::new();
        let half_of = |c: usize| if c <= h { c } else { ncol - c };
        let eps = 0.0008f32;
        let sections: Vec<[Section; 3]> = thetas
            .iter()
            .map(|&t| {
                [
                    band.section((t - eps).max(0.0)),
                    band.section(t),
                    band.section((t + eps).min(band.half)),
                ]
            })
            .collect();
        let belt_y: Vec<f32> = sections
            .iter()
            .map(|s3| band.point_in(&s3[1], 1.0).y)
            .collect();
        // --- Band -----------------------------------------------------------
        let mut ids = vec![0u32; ncol * (nrows + 1)];
        for r in 0..=nrows {
            let v = vrows[r];
            for c in 0..ncol {
                let cc = half_of(c);
                let sx = if c <= h { 1.0 } else { -1.0 };
                let s3 = &sections[cc];
                let mut p = band.point_in(&s3[1], v);
                let mut nn = band.normal_in(s3, v);
                p.x *= sx;
                nn.x *= sx;
                if cc == 0 || cc == h {
                    p.x = 0.0;
                    nn.x = 0.0;
                }
                let id = mesh.vert(p, nn, 0, 1.0);
                mesh.smooth[id as usize] = 1;
                ids[r * ncol + c] = id;
                arc.push(band.half - thetas[cc]);
                lid.push(false);
                ytop.push(belt_y[cc]);
                grid.push([c as u32, r as u32]);
                patch.push(0);
            }
        }
        for r in 0..nrows {
            for c in 0..ncol {
                let c1 = (c + 1) % ncol;
                mesh.quad(
                    ids[r * ncol + c],
                    ids[r * ncol + c1],
                    ids[(r + 1) * ncol + c1],
                    ids[(r + 1) * ncol + c],
                );
            }
        }
        let top: Vec<u32> = (0..ncol).map(|c| ids[nrows * ncol + c]).collect();
        // The lid frame per belt column (rear → front).
        let tr = prof.top_radius;
        let rows: Vec<LidRow> = (0..=h)
            .map(|c| {
                let e = mesh.pos[top[c] as usize];
                let z = e.z;
                LidRow {
                    z,
                    x_t: e.x,
                    y_t: e.y,
                    x_r: (prof.top_half_width.eval(z) - tr)
                        .clamp(0.0, (e.x - 0.04).max(0.0) * 0.95),
                    y_e: prof.top.eval(z).max(e.y + 0.005),
                }
            })
            .collect();
        // --- Ramp: belt → top line ------------------------------------------
        let topf = Top::new(&band);
        let cols: Vec<[Col; 3]> = thetas
            .iter()
            .map(|&t| [topf.col(t - eps), topf.col(t), topf.col(t + eps)])
            .collect();
        let knots: Vec<[f32; 7]> = cols.iter().map(|c3| c3[1].knots()).collect();

        // Rows per span (shoulder, ledge, fillet ×2, sail, rail round): enough
        // round the car for the span's length (the normals are the
        // surface's own, so a flat span can be coarse) and for its turn (a
        // row every 12° of slope).
        let span_need = |c: &Col, a: f32, b: f32| -> f32 {
            let mut len = 0.0;
            let mut q = c.at(a);
            for i in 1..=12 {
                let p = c.at(a + (b - a) * i as f32 / 12.0);
                len += p.distance(q);
                q = p;
            }
            let e = 0.0005;
            let slope = |s: f32| (c.h(s + e) - c.h(s - e)) / (2.0 * e);
            let turn = (slope(b - e).atan() - slope(a + e).atan()).abs();
            (len / (1.6 * step)).max(turn / (12f32 / super::detail().max(0.25)).to_radians())
        };
        let mut nspan = [1usize; 6];
        for (cc, c3) in cols.iter().enumerate() {
            let k = knots[cc];
            for sp in 0..6 {
                let n = span_need(&c3[1], k[sp], k[sp + 1]).ceil() as usize;
                nspan[sp] = nspan[sp].max(n);
            }
        }
        let mins = [2usize, 1, 2, 2, 1, 3];
        for sp in 0..6 {
            nspan[sp] = nspan[sp].max(super::segs(mins[sp], mins[sp]));
        }
        let nr_ramp: usize = nspan.iter().sum();
        // Per column the rows' s: by arc length within each span.
        let ramp_s: Vec<Vec<f32>> = cols
            .iter()
            .enumerate()
            .map(|(cc, c3)| {
                let c = &c3[1];
                let k = knots[cc];
                let mut out = vec![0.0f32];
                for sp in 0..6 {
                    let (a, b) = (k[sp], k[sp + 1]);
                    let ns = 16;
                    let mut acc = vec![0.0f32];
                    let mut q = c.at(a);
                    for i in 1..=ns {
                        let p = c.at(a + (b - a) * i as f32 / ns as f32);
                        acc.push(acc.last().unwrap() + p.distance(q));
                        q = p;
                    }
                    let total = *acc.last().unwrap();
                    let mut j = 0;
                    for r in 1..=nspan[sp] {
                        let target = total * r as f32 / nspan[sp] as f32;
                        while j + 1 < ns && acc[j + 1] < target {
                            j += 1;
                        }
                        let t =
                            ((target - acc[j]) / (acc[j + 1] - acc[j]).max(1e-9)).clamp(0.0, 1.0);
                        out.push(a + (b - a) * (j as f32 + t) / ns as f32);
                    }
                }
                out
            })
            .collect();
        let mut aid = vec![vec![0u32; ncol]; nr_ramp + 1];
        aid[0].copy_from_slice(&top);
        for j in 1..=nr_ramp {
            for c in 0..ncol {
                let cc = half_of(c);
                let sx = if c <= h { 1.0 } else { -1.0 };
                let c3 = &cols[cc];
                let s = ramp_s[cc][j];
                let mut p = c3[1].at(s);
                let mut nn = if j == nr_ramp {
                    topf.normal(p.x, p.z)
                } else {
                    let ds = 0.0006;
                    let t1 = c3[2].at(s) - c3[0].at(s);
                    let t2 = c3[1].at(s + ds) - c3[1].at(s - ds);
                    let n = t1.cross(t2).normalize_or_zero();
                    let out = p - Vec3::new(0.0, 0.55, p.z.clamp(1.0, l - 1.0));
                    if n.dot(out) < 0.0 {
                        -n
                    } else {
                        n
                    }
                };
                p.x *= sx;
                nn.x *= sx;
                if cc == 0 || cc == h {
                    p.x = 0.0;
                    nn.x = 0.0;
                }
                let id = mesh.vert(p, nn, 0, 1.0);
                mesh.smooth[id as usize] = 1;
                aid[j][c] = id;
                arc.push(band.half - thetas[cc]);
                lid.push(true);
                ytop.push(belt_y[cc]);
                grid.push([u32::MAX, u32::MAX]);
                patch.push(1);
            }
        }
        for j in 0..nr_ramp {
            for c in 0..ncol {
                let c1 = (c + 1) % ncol;
                mesh.quad(aid[j][c], aid[j][c1], aid[j + 1][c1], aid[j + 1][c]);
            }
        }
        // --- Top: rows anchored on the top line ------------------------------
        let ring = &aid[nr_ramp];
        let l1: Vec<Vec3> = (0..=h).map(|c| mesh.pos[ring[c] as usize]).collect();
        // Columns: centre → dome crease (with a fillet column either side) →
        // the top line, mirrored.
        let wd = 0.006f32;
        let crease_of = |i: usize| -> f32 {
            let x = l1[i].x.max(1e-3);
            style.crease_x(l1[i].z).min(0.5 * x)
        };
        // The top is gently curved and its normals are its own: half again
        // the grid spacing across.
        let (mut n1, mut n2) = (1usize, 1usize);
        for i in 1..h {
            let xc = crease_of(i);
            n1 = n1.max(((xc - wd).max(0.0) / (1.5 * step)).ceil() as usize);
            n2 = n2.max(((l1[i].x - xc - wd).max(0.0) / (1.5 * step)).ceil() as usize);
        }
        let xs_of = |i: usize| -> Vec<f32> {
            let x = l1[i].x.max(1e-3);
            let xc = crease_of(i);
            let wdi = wd.min(0.2 * xc);
            let mut right = Vec::with_capacity(n1 + n2 + 3);
            for k in 0..=n1 {
                right.push((xc - wdi) * k as f32 / n1 as f32);
            }
            right.push(xc);
            right.push(xc + wdi);
            for k in 1..=n2 {
                right.push(xc + wdi + (x - xc - wdi) * k as f32 / n2 as f32);
            }
            let mut row: Vec<f32> = right.iter().rev().map(|v| -v).collect();
            row.extend_from_slice(&right[1..]);
            row
        };
        let m = 2 * (n1 + n2 + 2);
        // Rows on every top-line vertex, with rows between them where those
        // are far apart along the car (their ends fan into the next
        // top-line vertex, so the mesh stays closed and manifold).
        let row_step = 0.75 * style.step_along;
        let mut tid: Vec<Vec<u32>> = Vec::with_capacity(2 * h + 1);
        let mut prev: Option<(Vec<f32>, f32)> = None;
        for i in 0..=h {
            let right = ring[i];
            let left = ring[(ncol - i) % ncol];
            if i == 0 || i == h {
                tid.push(vec![right; m + 1]);
                prev = None;
                continue;
            }
            let xs = xs_of(i);
            debug_assert_eq!(xs.len(), m + 1);
            let z = l1[i].z;
            let mut subs: Vec<(Vec<f32>, f32)> = Vec::new();
            if let Some((pxs, pz)) = &prev {
                let ns = ((z - pz) / row_step).ceil().max(1.0) as usize;
                for j in 1..ns {
                    let t = j as f32 / ns as f32;
                    let xs_j: Vec<f32> =
                        pxs.iter().zip(&xs).map(|(a, b)| a + (b - a) * t).collect();
                    subs.push((xs_j, pz + (z - pz) * t));
                }
            }
            subs.push((xs.clone(), z));
            for (xs_j, zj) in &subs {
                let mut row = Vec::with_capacity(m + 1);
                for (k, &x) in xs_j.iter().enumerate() {
                    if k == 0 {
                        row.push(left);
                        continue;
                    }
                    if k == m {
                        row.push(right);
                        continue;
                    }
                    let p = Vec3::new(x, topf.y(x, *zj), *zj);
                    let id = mesh.vert(p, topf.normal(x, *zj), 0, 1.0);
                    mesh.smooth[id as usize] = 1;
                    arc.push(band.half - thetas[i]);
                    lid.push(true);
                    ytop.push(belt_y[i]);
                    grid.push([u32::MAX, u32::MAX]);
                    patch.push(2);
                    row.push(id);
                }
                tid.push(row);
            }
            prev = Some((xs, z));
        }
        for i in 0..tid.len() - 1 {
            for k in 0..m {
                mesh.quad(tid[i][k], tid[i][k + 1], tid[i + 1][k + 1], tid[i + 1][k]);
            }
        }
        mesh.orient(|p| p - Vec3::new(0.0, 0.55, p.z.clamp(1.0, l - 1.0)));
        Self {
            mesh,
            arc,
            lid,
            ytop,
            grid,
            patch,
            band,
            rows,
            top_edge: top,
            cols: ncol,
            band_rows: nrows,
        }
    }

    /// The lid frame at z (interpolated between rows; rows run rear → front).
    pub fn lid_row(&self, z: f32) -> LidRow {
        let r = &self.rows;
        let i = r.partition_point(|q| q.z < z).clamp(1, r.len() - 1);
        let (a, b) = (r[i - 1], r[i]);
        let t = ((z - a.z) / (b.z - a.z).max(1e-6)).clamp(0.0, 1.0);
        let l = |x: f32, y: f32| x + (y - x) * t;
        LidRow {
            z,
            x_t: l(a.x_t, b.x_t),
            y_t: l(a.y_t, b.y_t),
            x_r: l(a.x_r, b.x_r),
            y_e: l(a.y_e, b.y_e),
        }
    }

    /// The belt line at z (side view): the band's top edge, kicked up behind
    /// the cabin along the deck ledge as it falls from the deck to the belt
    /// (the feature line the quarter light sits on).
    pub fn belt(&self, z: f32) -> f32 {
        let r = self.lid_row(z);
        let [zd0, zd1, _, fw] = self.band.style.rear_deck;
        if fw <= 0.0 || z >= zd1 {
            return r.y_t;
        }
        r.y_t.max(ledge_y(&self.band.prof, zd0, zd1, z, r.y_t))
    }

    /// The band's top-edge height at arc `a` from the front centre.
    pub fn band_top_y(&self, a: f32) -> f32 {
        let th = self.band.theta_front(a.max(0.0)).clamp(0.0, self.band.half);
        self.band.section(th).yt
    }

    /// The arc from the front centre of the side point at station z.
    pub fn arc_at_z(&self, z: f32) -> f32 {
        let (mut lo, mut hi) = (0.0f32, self.band.half);
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            if self.band.plan(m).0.y < z {
                lo = m;
            } else {
                hi = m;
            }
        }
        self.band.half - 0.5 * (lo + hi)
    }

    /// A band point and its normal at arc `a` from the front centre (side +1
    /// left, −1 right) and height y.
    pub fn band_at(&self, a: f32, y: f32, side: f32) -> (Vec3, Vec3) {
        let th = self.band.theta_front(a.max(0.0)).clamp(0.0, self.band.half);
        let sc = self.band.section(th);
        let v = self.band.v_at(&sc, y);
        let p = self.band.point_in(&sc, v);
        let e = 0.002;
        let pv = self.band.point_in(&sc, (v + e).min(1.0));
        let pm = self.band.point_in(&sc, (v - e).max(0.0));
        let th2 = (th + e).min(self.band.half);
        let th1 = (th - e).max(0.0);
        let pa = self.band.point(th2, v);
        let pb = self.band.point(th1, v);
        let mut n = (pa - pb).cross(pv - pm).normalize_or_zero();
        let out = Vec3::new(sc.n.x, 0.0, sc.n.y);
        if n.dot(out) < 0.0 {
            n = -n;
        }
        let mut p = p;
        if side < 0.0 {
            p.x = -p.x;
            n.x = -n.x;
        }
        (p, n)
    }
}
