//! The body surface: one connected, dense triangle mesh that the panels are cut
//! from.
//!
//! * **Plan outline.** The widest line of the body (the shoulder crease on the
//!   sides) seen from above: the profile's half width with the nose and tail
//!   bowed and the corners rounded (a smooth intersection), sampled by arc
//!   length θ from the rear centre round the left side to the front centre.
//! * **Band.** The lower body swept round that outline: at every θ a section
//!   curve in (offset along the outline's normal, height): a tuck-under curl at
//!   the bottom, a lower body rising to the crease (the widest point, a crisp
//!   change of slope), and an upper body leaning in to the shoulder. The section
//!   blends between side, nose and tail shapes with the outline's direction, so
//!   bumpers wrap round the corners without end caps. Arch flares swell the
//!   sides round the wheels. The crease and the lower line sit on fixed rows, so
//!   their highlights run straight along the grid.
//! * **Lid.** Everything above the band's top edge (bonnet, windscreen, roof,
//!   pillars, side glass, backlight, deck) as a height field over plan view:
//!   the edge height, rising over a ramp (the greenhouse's tumblehome, the fender
//!   tops) to the profile's top line with its crown. The ramp is measured as the
//!   plan distance to the edge, so the bonnet rises from its leading edge as
//!   naturally as from its sides. Rows run across the car at constant z; their
//!   ends are the band's top vertices (one mesh, no crack).
//!
//! Coordinates: definition coordinates (x across, +x the car's left; y up; z
//! from the rear end, 0, to the front, `length`).

use super::{hermite, smax, smoothstep, Mesh};
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
    /// Grid spacing (m) across the sections (band rows, lid columns) and
    /// along the car (band columns, lid rows): the body curves far more across
    /// than along, and every cut line is exact whatever the grid.
    pub step: f32,
    pub step_along: f32,
    /// The lid's roll-over length (m) from its edge at the nose and at the
    /// tail (the sides keep the tumblehome ramp): short gives a crisp bonnet
    /// leading edge and a crisp boot deck.
    pub nose_ramp: f32,
    pub tail_ramp: f32,
    /// The rear deck as its own surface: a flat, crowned lid at the profile's
    /// top behind z = `rear_deck[0]`, its level carried forward (falling to
    /// the shoulder by z = `rear_deck[1]`) as the quarters' shoulder, rolled
    /// over its edges in `rear_deck[2]` m. The lid is the smooth max (radius
    /// `rear_deck[3]` m) of it and the greenhouse surface, so the C-pillar
    /// rises out of the deck along a defined crease. Zero length: none.
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
}

/// Smooth max(s, 0) with a rounding of w (m).
fn soft_pos(s: f32, w: f32) -> f32 {
    0.5 * (s + (s * s + w * w).sqrt())
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

    /// Detail heights (m) added to the lid at plan (x, z) after it is relaxed:
    /// the bonnet's power dome and the boot lip. `z_tail` is the tail edge's
    /// z at this x.
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
    /// Per vertex: arc distance (m) round the plan outline from the front centre
    /// (band; the nearest edge point's for the lid), and whether it is on the lid.
    pub arc: Vec<f32>,
    pub lid: Vec<bool>,
    /// Per vertex: the band's top-edge height at its place round the car (the
    /// arc view's reference height).
    pub ytop: Vec<f32>,
    /// Per vertex: grid (column, row) of band vertices (u32::MAX on the lid).
    pub grid: Vec<[u32; 2]>,
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
}

#[derive(Clone, Copy, Debug)]
pub struct Section {
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
            // Walking rear → front along +x: outward is (t.z, −t.x)·… pick the one
            // pointing away from the centre.
            let mut nn = Vec2::new(t.y, -t.x);
            if nn.dot(samples[i] - c) < 0.0 {
                nn = -nn;
            }
            outline.push((samples[i], nn));
        }
        Self {
            prof: prof.clone(),
            style: style.clone(),
            outline,
            half,
            v_lower: 0.0,
            v_crease: 0.0,
            v_scoop: 0.0,
        }
    }

    /// Plan point and outward normal at θ ∈ [0, half] (x ≥ 0 side).
    pub fn plan(&self, theta: f32) -> (Vec2, Vec2) {
        let n = self.outline.len() - 1;
        let u = (theta / self.half).clamp(0.0, 1.0) * n as f32;
        let i = (u.floor() as usize).min(n - 1);
        let t = u - i as f32;
        let (p0, n0) = self.outline[i];
        let (p1, n1) = self.outline[i + 1];
        (p0.lerp(p1, t), n0.lerp(n1, t).normalize_or_zero())
    }

    /// The section at θ.
    pub fn section(&self, theta: f32) -> Section {
        let (p, n) = self.plan(theta);
        let st = &self.style;
        let wf = smoothstep(0.25, 0.97, n.y);
        let wr = smoothstep(0.25, 0.97, -n.y);
        let mut side = st.side;
        side.crease = st.crease.eval(p.y);
        // Fenders and deck sides roll toward the flat lid; the cabin's doors
        // lean in less (the glass continues them).
        let rise = (self.prof.top.eval(p.y) - self.prof.shoulder.eval(p.y)).max(0.0);
        let roll = 1.0 - smoothstep(0.08, 0.30, rise);
        // Not behind the cabin when the rear deck is its own surface: there
        // the quarter stays upright into the deck's tight roll (a hard
        // shoulder).
        let front = if st.rear_deck[2] > 0.0 {
            smoothstep(st.rear_deck[1], st.rear_deck[1] + 0.6, p.y)
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
        let ys = st
            .scoop
            .eval(p.y)
            .clamp(yl + 0.3 * (yc - yl), yl + 0.75 * (yc - yl));
        Section {
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
            let d = hermite(t, s.d_lower, 0.0, 0.0, s.a_below * dy);
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
                (self.style.rear_flare, self.style.flare_width)
            } else {
                (self.style.flare, self.style.flare_width)
            };
            // A swell that rises over the flare width to a plateau at the lip.
            let k = smoothstep(w.radius + width, w.radius + 0.03, r);
            best = best.max(amt * k);
        }
        best
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
            - side_w * sidish * self.style.scoop_at(sc.p.y, q.y, sc.yl, sc.ys);
        Vec3::new(sc.p.x + sc.n.x * d, q.y, sc.p.y + sc.n.y * d)
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

impl Lut {
    pub fn eval(&self, z: f32) -> f32 {
        let u = ((z - self.z0) / self.dz).clamp(0.0, (self.v.len() - 1) as f32);
        let i = (u.floor() as usize).min(self.v.len() - 2);
        let t = u - i as f32;
        self.v[i] + (self.v[i + 1] - self.v[i]) * t
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

/// The profile with its curves blurred along the car (Gaussian, σ m): the
/// package curves are monotone cubics through a handful of points, smooth in
/// value and slope but with curvature jumps at the points, which a clearcoat's
/// reflections show as ripples.
pub fn smooth_profile(p: &BodyProfile, sigma: f32) -> Pkg {
    let l = p.length;
    let dz = 0.01f32;
    let n = (l / dz).ceil() as usize;
    let blur = |c: &Curve| -> Lut {
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
    };
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
        // side of the crease and of the lower-door feature line, so the smooth
        // normals turn within a tight fillet there (a crisp line of light)
        // instead of over a whole row.
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
        let h = (band.half / style.step_along).ceil() as usize;
        let ncol = 2 * h;
        let mut mesh = Mesh::default();
        let mut arc = Vec::new();
        let mut lid = Vec::new();
        let mut ytop: Vec<f32> = Vec::new();
        let mut grid = Vec::new();
        // Band vertices: row-major, columns round the loop.
        let theta_of = |c: usize| -> (f32, f32) {
            if c <= h {
                (band.half * c as f32 / h as f32, 1.0)
            } else {
                (band.half * (ncol - c) as f32 / h as f32, -1.0)
            }
        };
        let sections: Vec<Section> = (0..=h)
            .map(|c| band.section(band.half * c as f32 / h as f32))
            .collect();
        let mut ids = vec![0u32; ncol * (nrows + 1)];
        for r in 0..=nrows {
            let v = vrows[r];
            for c in 0..ncol {
                let (theta, sx) = theta_of(c);
                let sc = &sections[if c <= h { c } else { ncol - c }];
                let mut p = band.point_in(sc, v);
                p.x *= sx;
                if c == 0 || c == h {
                    p.x = 0.0;
                }
                let id = mesh.vert(p, Vec3::Y, 0, 1.0);
                mesh.smooth[id as usize] = 1;
                ids[r * ncol + c] = id;
                arc.push(band.half - theta);
                lid.push(false);
                grid.push([c as u32, r as u32]);
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
        for r in 0..=nrows {
            for c in 0..ncol {
                ytop.push(mesh.pos[top[c] as usize].y);
                let _ = r;
            }
        }
        // Lid: rows i = 0..=h at the band's top-edge points.
        let edge: Vec<Vec3> = (0..=h).map(|c| mesh.pos[top[c] as usize]).collect();
        let edge_arc: Vec<f32> = (0..=h)
            .map(|c| band.half - band.half * c as f32 / h as f32)
            .collect();
        let tr = prof.top_radius;
        let mut rows = Vec::with_capacity(h + 1);
        for e in &edge {
            let z = e.z;
            let x_r = (prof.top_half_width.eval(z) - tr).clamp(0.0, (e.x - 0.04).max(0.0) * 0.95);
            let y_e = prof.top.eval(z).max(e.y + 0.005);
            rows.push(LidRow {
                z,
                x_t: e.x,
                y_t: e.y,
                x_r,
                y_e,
            });
        }
        // Lid height at plan (x, z): the edge height at the nearest edge point,
        // ramped up to the top line by the plan distance from the edge.
        let ramp_of = |z: f32| -> f32 {
            (prof.half_width.eval(z) + prof.side_bulge - (prof.top_half_width.eval(z) - tr))
                .max(0.08)
        };
        // Per edge point: how much it faces forward (nose) or back (tail): the
        // lid rolls over its leading and trailing edges in a short, crisp ramp.
        let edge_ramp: Vec<f32> = edge
            .iter()
            .enumerate()
            .map(|(c, e)| {
                let (_, n) = band.plan(band.half * c as f32 / h as f32);
                let wf = smoothstep(0.35, 0.92, n.y);
                let wr = smoothstep(0.35, 0.92, -n.y);
                let side = ramp_of(e.z);
                side + (style.nose_ramp - side) * wf + (style.tail_ramp - side) * wr
            })
            .collect();
        let lid_y = |x: f32, z: f32, near: usize| -> (f32, f32) {
            // Nearest edge point (plan), searching near `near`.
            let lo = near.saturating_sub(90);
            let hi = (near + 90).min(h);
            let q = Vec2::new(x.abs(), z);
            let mut best = (f32::MAX, edge[near].y, edge_arc[near], edge_ramp[near]);
            for k in lo..hi {
                let a = Vec2::new(edge[k].x, edge[k].z);
                let b = Vec2::new(edge[k + 1].x, edge[k + 1].z);
                let ab = b - a;
                let t = ((q - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                let d = (a + ab * t).distance(q);
                if d < best.0 {
                    let y = edge[k].y + (edge[k + 1].y - edge[k].y) * t;
                    let ar = edge_arc[k] + (edge_arc[k + 1] - edge_arc[k]) * t;
                    let rp = edge_ramp[k] + (edge_ramp[k + 1] - edge_ramp[k]) * t;
                    best = (d, y, ar, rp);
                }
            }
            let (d, yb, ar, ramp) = best;
            // Top line at z: the profile's top with its crown.
            let x_r = (prof.top_half_width.eval(z) - tr).max(0.05);
            let cabin = smoothstep(0.55, 0.95, z)
                * (1.0 - smoothstep(prof.length - 1.35, prof.length - 0.95, z));
            let crown_amt = style.lid_crown + (prof.crown - style.lid_crown) * cabin;
            let u = (x.abs() / x_r).min(1.0);
            let top = prof.top.eval(z) + crown_amt * (1.0 - u * u) * (1.0 - u * u);
            let top = top.max(yb + 0.004);
            let rise = top - yb;
            let t = (d / ramp).clamp(0.0, 1.0);
            // A straight-ish ramp (the glass) into a rounded rail; a long gentle
            // one when the rise is small (fender tops into the bonnet).
            let tau = (0.7 - (rise - 0.05) * 1.6).clamp(0.22, 0.7);
            let m = 1.0 / (1.0 - 0.5 * tau);
            let hh = if t <= 1.0 - tau {
                m * t
            } else {
                1.0 - m * (1.0 - t) * (1.0 - t) / (2.0 * tau)
            };
            let mut y = yb + rise * hh;
            // The rear deck (a crisp lid and the quarters' shoulder), merged
            // with the greenhouse by a smooth max: the C-pillar's crease.
            let [zd0, zd1, dr, k] = style.rear_deck;
            if dr > 0.0 && z < zd1 + 0.05 {
                let s = smoothstep(zd0, zd1, z);
                let deck0 = prof.top.eval(z.min(zd0));
                let deck = deck0 + style.lid_crown * (1.0 - s) * (1.0 - u * u) * (1.0 - u * u);
                let yd = deck + (yb - 0.01 - deck) * s;
                let td = (d / ramp.min(dr)).clamp(0.0, 1.0);
                // A short wall (the quarter's upper face) into a tight round.
                let tau_d = 0.45;
                let md = 1.0 / (1.0 - 0.5 * tau_d);
                let hd = if td <= 1.0 - tau_d {
                    md * td
                } else {
                    1.0 - md * (1.0 - td) * (1.0 - td) / (2.0 * tau_d)
                };
                let dy = yb + (yd - yb).max(0.0) * hd;
                y = smax(y, dy, k);
            }
            (y, ar)
        };
        // Columns per row: uniform by arc length across.
        let fine = 240usize;
        let mut row_pts: Vec<Vec<(Vec3, f32)>> = Vec::with_capacity(h + 1);
        let mut max_len = 0.0f32;
        for (i, row) in rows.iter().enumerate() {
            let mut line = Vec::with_capacity(fine + 1);
            for k in 0..=fine {
                let x = -row.x_t + 2.0 * row.x_t * k as f32 / fine as f32;
                let (y, ar) = if k == 0 || k == fine {
                    (row.y_t, edge_arc[i])
                } else {
                    lid_y(x, row.z, i)
                };
                line.push((Vec3::new(x, y, row.z), ar));
            }
            let len: f32 = line.windows(2).map(|w| w[0].0.distance(w[1].0)).sum();
            max_len = max_len.max(len);
            row_pts.push(line);
        }
        let m = ((max_len / step).ceil() as usize).max(8);
        let crease_x = |z: f32| -> f32 {
            let [xn, xc, _, _] = style.hood_crease;
            let [zc, zn] = style.hood_z;
            let t = ((z - zc) / (zn - zc)).clamp(0.0, 1.0);
            xc + (xn - xc) * t
        };
        let mut lid_ids: Vec<Vec<u32>> = Vec::with_capacity(h + 1);
        for (i, line) in row_pts.iter().enumerate() {
            let right = top[i];
            let left = top[(ncol - i) % ncol];
            if i == 0 || i == h {
                lid_ids.push(vec![right; m + 1]);
                continue;
            }
            let mut acc = vec![0.0f32];
            for w in line.windows(2) {
                acc.push(acc.last().unwrap() + w[0].0.distance(w[1].0));
            }
            let total = *acc.last().unwrap();
            // Columns uniform by arc length in three spans, split where the
            // bonnet's creases run (continued at the cowl's spacing behind
            // it): a column lies on each crease, so it reads as a straight line.
            let x_t = rows[i].x_t;
            let xl = crease_x(rows[i].z).min(0.5 * x_t);
            let s_at = |x: f32| -> f32 {
                let kf = ((x + x_t) / (2.0 * x_t) * fine as f32).clamp(0.0, fine as f32);
                let k0 = (kf.floor() as usize).min(fine - 1);
                acc[k0] + (acc[k0 + 1] - acc[k0]) * (kf - k0 as f32)
            };
            let (s_a, s_b) = (s_at(-xl), s_at(xl));
            let k1 = ((m as f32 * 0.34).round() as usize).clamp(1, m / 2 - 1);
            let k2 = m - k1;
            let mut row = Vec::with_capacity(m + 1);
            let mut j = 0;
            for k in 0..=m {
                if k == 0 {
                    row.push(left);
                    continue;
                }
                if k == m {
                    row.push(right);
                    continue;
                }
                let s = if k <= k1 {
                    s_a * k as f32 / k1 as f32
                } else if k <= k2 {
                    s_a + (s_b - s_a) * (k - k1) as f32 / (k2 - k1) as f32
                } else {
                    s_b + (total - s_b) * (k - k2) as f32 / (m - k2) as f32
                };
                while j + 2 < acc.len() && acc[j + 1] < s {
                    j += 1;
                }
                let t = ((s - acc[j]) / (acc[j + 1] - acc[j]).max(1e-9)).clamp(0.0, 1.0);
                let p = line[j].0.lerp(line[j + 1].0, t);
                let ar = line[j].1 + (line[j + 1].1 - line[j].1) * t;
                let id = mesh.vert(p, Vec3::Y, 0, 1.0);
                mesh.smooth[id as usize] = 1;
                arc.push(ar);
                lid.push(true);
                ytop.push(rows[i].y_t);
                grid.push([u32::MAX, u32::MAX]);
                row.push(id);
            }
            lid_ids.push(row);
        }
        // Relax the lid's heights (a few Jacobi passes of a Laplacian, edges
        // fixed): the distance ramp creases where two edges are equally near
        // (the bonnet's and deck's corners); this irons that out.
        for _ in 0..8 {
            let mut ny: Vec<(u32, f32)> = Vec::new();
            for i in 2..h.saturating_sub(1) {
                for k in 1..m {
                    let c = lid_ids[i][k];
                    let nb = [
                        lid_ids[i - 1][k],
                        lid_ids[i + 1][k],
                        lid_ids[i][k - 1],
                        lid_ids[i][k + 1],
                    ];
                    let avg = nb.iter().map(|&v| mesh.pos[v as usize].y).sum::<f32>() / 4.0;
                    // Not on the rear deck: its edges and crease are designed.
                    let [_, zd1, dr, _] = style.rear_deck;
                    let w = if dr > 0.0 {
                        smoothstep(zd1, zd1 + 0.3, mesh.pos[c as usize].z)
                    } else {
                        1.0
                    };
                    let y0 = mesh.pos[c as usize].y;
                    ny.push((c, y0 + w * (avg - y0)));
                }
            }
            for (c, y) in ny {
                let p = &mut mesh.pos[c as usize].y;
                *p += 0.5 * (y - *p);
            }
        }
        // Details on top of the relaxed lid (the relaxation would blur their
        // creases): the bonnet's power dome and the boot lip.
        let tail_z = |x: f32| -> f32 {
            // The tail edge's z at this x (the band's outline, rear half).
            let mut best = (f32::MAX, 0.0f32);
            for e in edge.iter().take(h / 2 + 1) {
                let dx = (e.x - x.abs()).abs();
                if dx < best.0 {
                    best = (dx, e.z);
                }
            }
            best.1
        };
        for row in lid_ids.iter().take(h).skip(1) {
            for &v in row.iter().take(m).skip(1) {
                let p = mesh.pos[v as usize];
                mesh.pos[v as usize].y += style.lid_detail(p.x, p.z, tail_z(p.x));
            }
        }
        for i in 0..h {
            for k in 0..m {
                mesh.quad(
                    lid_ids[i][k],
                    lid_ids[i][k + 1],
                    lid_ids[i + 1][k + 1],
                    lid_ids[i + 1][k],
                );
            }
        }
        let l = prof.length;
        mesh.orient(|p| p - Vec3::new(0.0, 0.55, p.z.clamp(1.0, l - 1.0)));
        mesh.face_normals();
        // Rows for lookups use the band's x ≥ 0 order (rear → front).
        let _ = &mut rows;
        Self {
            mesh,
            arc,
            lid,
            ytop,
            grid,
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
