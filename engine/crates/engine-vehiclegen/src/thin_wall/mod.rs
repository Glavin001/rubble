//! A thin-walled section calculator: a member's cross-section as straight walls, and what
//! follows from it for a crash structure (`engine/docs/section-matching.md`).
//!
//! A [`ThinSection`] is one or more walls, each a midline polyline (open or closed) with a
//! thickness: a box rail, a C-channel, a hat on its closing plate, two stampings spot-welded
//! (one wall each; where they overlap at a flange the caller gives both, stacked). Each wall
//! becomes quadrilateral pieces that meet on their mitre lines ([`geometry::wall_pieces`]),
//! so a corner's material is counted once and a closed rectangular midline is exactly a
//! rectangular tube. Every integral is then exact for those polygons (Green's theorem), and
//! so is the plastic neutral axis (bisection on exact clipped areas). The only modelling
//! choice is the polygon itself: a curved wall is approximated by its chords.
//!
//! From the section: area, centroid, second moments (both axes, product, principal),
//! elastic section moduli, plastic moduli about any axis, and with a [`Material`] the
//! squash load σ_y·A, plastic moments σ_y·Z, axial stiffness EA/L and bending rigidity EI.
//! Folding and collapse formulas from the thin-walled crash literature are in [`crush`],
//! each with its source and how sure we are of its coefficients.
//!
//! Units: metres, pascals, newtons, kilograms.

pub mod crush;
pub mod geometry;
#[cfg(test)]
mod tests;

use geometry::{clip, wall_pieces, Moments, P2};
use serde::Serialize;

/// One wall: its midline and thickness (m). `closed`: the last point joins the first.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Wall {
    pub points: Vec<P2>,
    pub t: f64,
    pub closed: bool,
}

/// A cross-section: walls (one per stamping), in the section's plane (x across, y up).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ThinSection {
    pub walls: Vec<Wall>,
}

/// A steel (or other metal) for the section's strengths (Pa, kg/m³).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Material {
    pub e: f64,
    pub yield_pa: f64,
    pub ultimate_pa: f64,
    /// The flow stress a folding wall works at (the mean of its stress over the strains of a
    /// fold); None: (σ_y + σ_u) / 2, the usual estimate.
    pub flow_pa: Option<f64>,
    pub density: f64,
}

impl Material {
    pub fn steel(yield_pa: f64, ultimate_pa: f64) -> Self {
        Self { e: 210e9, yield_pa, ultimate_pa, flow_pa: None, density: 7850.0 }
    }
    pub fn flow(&self) -> f64 {
        self.flow_pa.unwrap_or(0.5 * (self.yield_pa + self.ultimate_pa))
    }
}

impl From<crate::structure::Grade> for Material {
    /// A generator grade's yield and ultimate strengths; its flow stress is the grade's own
    /// ([`crate::structure::Grade::flow_pa`], the same mean).
    fn from(g: crate::structure::Grade) -> Self {
        Self { flow_pa: Some(g.flow_pa()), ..Self::steel(g.yield_pa(), g.ultimate_pa()) }
    }
}

/// A section's geometric properties (m, m², m³, m⁴; angles in radians).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Properties {
    pub area: f64,
    pub centroid: P2,
    /// ∫(y − ȳ)² dA: bending about the horizontal (x) axis through the centroid.
    pub ixx: f64,
    /// ∫(x − x̄)² dA: bending about the vertical (y) axis.
    pub iyy: f64,
    /// ∫(x − x̄)(y − ȳ) dA.
    pub ixy: f64,
    /// Principal second moments (i1 ≥ i2) and the angle of the axis of i1 from x.
    pub i1: f64,
    pub i2: f64,
    pub principal_angle: f64,
    /// Elastic section moduli I / c (c the farthest fibre from the axis).
    pub wx: f64,
    pub wy: f64,
    /// Plastic section moduli about the x and y directions (equal-area axes).
    pub zx: f64,
    pub zy: f64,
    /// The plastic neutral axes: y of the one for zx, x of the one for zy.
    pub pna_y: f64,
    pub pna_x: f64,
}

impl ThinSection {
    /// A rectangular tube `b` across by `h` high (outer, m), wall `t`: one closed wall on the
    /// midline (b − t) × (h − t), centred on the origin.
    pub fn rect_tube(b: f64, h: f64, t: f64) -> Self {
        let (x, y) = (0.5 * (b - t), 0.5 * (h - t));
        Self { walls: vec![Wall { points: vec![[-x, -y], [x, -y], [x, y], [-x, y]], t, closed: true }] }
    }
    /// A circular tube of outer diameter `d`, wall `t`: a closed wall of `n` chords whose
    /// vertices lie on the midline circle. Its area and moments converge on the tube's as
    /// (π/n)²/6 (n = 20,000: 4e-9).
    pub fn circular_tube(d: f64, t: f64, n: usize) -> Self {
        let r = 0.5 * (d - t);
        let points = (0..n)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / n as f64;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        Self { walls: vec![Wall { points, t, closed: true }] }
    }
    /// A C-channel `h` high by `b` wide (outer), web and flanges `t`, web on the left
    /// (x = 0 outside), flanges pointing +x.
    pub fn c_channel(h: f64, b: f64, t: f64) -> Self {
        let (xw, yf) = (0.5 * t, 0.5 * (h - t));
        Self { walls: vec![Wall { points: vec![[b, -yf], [xw, -yf], [xw, yf], [b, yf]], t, closed: false }] }
    }
    /// A top hat `w` wide over its webs, `h` high (outer), flanges `fl` outstanding each side,
    /// all of gauge `t`, open side down, the flanges' underside at y = 0; with `plate`, a
    /// closing plate of gauge `t_plate` spot-welded under the flanges (a second wall).
    pub fn hat(w: f64, h: f64, fl: f64, t: f64, plate: Option<f64>) -> Self {
        let (x, yt, yb) = (0.5 * (w - t), h - 0.5 * t, 0.5 * t);
        let mut walls = vec![Wall { points: vec![[-x - fl, yb], [-x, yb], [-x, yt], [x, yt], [x, yb], [x + fl, yb]], t, closed: false }];
        if let Some(tp) = plate {
            walls.push(Wall { points: vec![[-x - fl - 0.5 * t, -0.5 * tp], [x + fl + 0.5 * t, -0.5 * tp]], t: tp, closed: false });
        }
        Self { walls }
    }

    /// Every wall's pieces (quadrilaterals).
    pub fn pieces(&self) -> Result<Vec<[P2; 4]>, String> {
        let mut out = Vec::new();
        for (i, w) in self.walls.iter().enumerate() {
            out.extend(wall_pieces(&w.points, w.t, w.closed).map_err(|e| format!("wall {i}: {e}"))?);
        }
        Ok(out)
    }

    pub fn properties(&self) -> Result<Properties, String> {
        let pieces = self.pieces()?;
        let m = pieces.iter().fold(Moments::default(), |m, p| m.add(Moments::of(p)));
        if !(m.a > 0.0) {
            return Err("the section has no area".into());
        }
        let (cx, cy) = (m.sx / m.a, m.sy / m.a);
        let ixx = m.ixx_o - m.a * cy * cy;
        let iyy = m.iyy_o - m.a * cx * cx;
        let ixy = m.ixy_o - m.a * cx * cy;
        let mean = 0.5 * (ixx + iyy);
        let r = (0.25 * (ixx - iyy).powi(2) + ixy * ixy).sqrt();
        let principal_angle = 0.5 * (2.0 * ixy).atan2(ixx - iyy);
        let far = |f: &dyn Fn(P2) -> f64| pieces.iter().flatten().map(|p| f(*p).abs()).fold(0.0, f64::max);
        let (pna_y, zx) = plastic(&pieces, [0.0, 1.0], m.a);
        let (pna_x, zy) = plastic(&pieces, [1.0, 0.0], m.a);
        Ok(Properties {
            area: m.a,
            centroid: [cx, cy],
            ixx,
            iyy,
            ixy,
            i1: mean + r,
            i2: mean - r,
            principal_angle,
            wx: ixx / far(&|p| p[1] - cy),
            wy: iyy / far(&|p| p[0] - cx),
            zx,
            zy,
            pna_y,
            pna_x,
        })
    }

    /// Bending about an axis at `angle` from x (rad): the offset of its equal-area axis
    /// along the axis's normal (−sin, cos), and the plastic modulus Z.
    pub fn plastic_modulus_at(&self, angle: f64) -> Result<(f64, f64), String> {
        let pieces = self.pieces()?;
        let a = pieces.iter().map(|p| Moments::of(p).a).sum();
        Ok(plastic(&pieces, [-angle.sin(), angle.cos()], a))
    }
}

/// The equal-area axis normal to `n` (its offset s along n) and the plastic modulus
/// ∫|n·p − s| dA, by bisection on the exact area on each side.
fn plastic(pieces: &[[P2; 4]], n: P2, area: f64) -> (f64, f64) {
    let side = |s: f64| -> Moments { pieces.iter().fold(Moments::default(), |m, p| m.add(Moments::of(&clip(p, n, s)))) };
    let proj = |p: &P2| n[0] * p[0] + n[1] * p[1];
    let (mut lo, mut hi) = pieces.iter().flatten().fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(proj(p)), hi.max(proj(p))));
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if side(mid).a > 0.5 * area {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let s = 0.5 * (lo + hi);
    let up = side(s);
    let all = pieces.iter().fold(Moments::default(), |m, p| m.add(Moments::of(p)));
    // first moments of each side about the line n·p = s
    let q_up = n[0] * up.sx + n[1] * up.sy - s * up.a;
    let (dn_a, dn_s) = (all.a - up.a, n[0] * (all.sx - up.sx) + n[1] * (all.sy - up.sy));
    let q_dn = s * dn_a - dn_s;
    (s, q_up + q_dn)
}

/// What a section in a material gives a member of length `len` (N, N/m, N·m², N·m).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Strengths {
    /// σ_y·A
    pub squash_n: f64,
    /// σ_u·A
    pub ultimate_n: f64,
    /// EA / L
    pub axial_k: f64,
    /// E·Ixx, E·Iyy
    pub eixx: f64,
    pub eiyy: f64,
    /// σ_y·W: first yield in bending
    pub my_x: f64,
    pub my_y: f64,
    /// σ_y·Z: the fully plastic moments
    pub mp_x: f64,
    pub mp_y: f64,
    /// ρA (kg/m)
    pub mass_per_m: f64,
}

impl Properties {
    pub fn strengths(&self, m: &Material, len: f64) -> Strengths {
        Strengths {
            squash_n: m.yield_pa * self.area,
            ultimate_n: m.ultimate_pa * self.area,
            axial_k: m.e * self.area / len,
            eixx: m.e * self.ixx,
            eiyy: m.e * self.iyy,
            my_x: m.yield_pa * self.wx,
            my_y: m.yield_pa * self.wy,
            mp_x: m.yield_pa * self.zx,
            mp_y: m.yield_pa * self.zy,
            mass_per_m: m.density * self.area,
        }
    }
}
