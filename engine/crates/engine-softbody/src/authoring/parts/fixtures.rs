//! Parametric part tools. Each takes a few design numbers (and, for parts set
//! into the body, the body surface and the opening's distance field) and
//! returns a modelled mesh with its materials:
//!
//! * [`lamp`]: a lamp set into a body opening: a lens that continues the body's
//!   surface, a housing extruded back along the lamp's axis, a reflector back,
//!   projector bores and a light guide (DRL or tail bars) that follows the
//!   outline;
//! * [`grille`]: a recessed grille: chrome surround, dark throat, back mesh and
//!   slats across the opening;
//! * [`wheel`]: tyre (tread grooves, shoulders, sidewall with a rim protector
//!   and a lettering band), a spoked rim with real spoke walls, barrel, lip,
//!   centre cap and lug nuts, the brake disc and hat; [`caliper`] separately
//!   (it does not turn);
//! * [`mirror`]: housing on an arm, with its glass;
//! * [`handle`]: a pull handle over a dark pocket;
//! * [`arch_liner`], [`underbody`], [`exhaust`].

use super::trim::{self, EdgeStyle};
use super::{smoothstep, Mesh};
use crate::authoring::skin::mat;
use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};

/// A lamp set into a body opening.
#[derive(Clone, Debug)]
pub struct LampSpec {
    /// Into the car (the housing's extrusion), unit.
    pub axis: Vec3,
    pub depth: f32,
    pub lens: u8,
    pub housing: u8,
    pub reflector: u8,
    /// Light guides: (inner level, outer level) bands of the opening's distance
    /// field (negative inside), set back `guide_depth` behind the lens.
    pub guides: Vec<(f32, f32, u8)>,
    pub guide_depth: f32,
    /// Projector bores: centres (on the body surface) and radius.
    pub projectors: Vec<Vec3>,
    pub projector_r: f32,
    /// When not empty, the guides are kept only above this line (points on
    /// the body, ordered): a DRL blade along the lamp's upper edge and round
    /// its tip rather than a ring.
    pub guide_above: Vec<Vec3>,
    /// The lens swells this far (m) proud of the body toward its middle (a
    /// three-dimensional lens rather than a flush window).
    pub lens_bulge: f32,
    /// An inner graphic: horizontal light blades at these fractions of the
    /// lens's height (world y), `bar_half` m half high, set back behind the
    /// guides, kept 1 cm inside the outline.
    pub bars: Vec<f32>,
    pub bar_half: f32,
}

/// Build a lamp from the body mesh `m` and the opening field `f` (metric,
/// negative inside the opening).
pub fn lamp(m: &Mesh, f: &[f32], spec: &LampSpec) -> Mesh {
    let mut out = Mesh::default();
    // Lens: the body surface over the opening (a little beyond it, under the
    // body's rolled edge), set 2 mm in.
    let lens_src = trim::clip(m, f, 0.003);
    if lens_src.mesh.is_empty() {
        return out;
    }
    let mut lens = trim::offset(&lens_src.mesh, -0.002, false, Some(spec.lens), 1.0);
    if spec.lens_bulge > 0.0 {
        for k in 0..lens.len() {
            let w = smoothstep(0.0, -0.03, lens_src.field[k]);
            lens.pos[k] += lens_src.mesh.nrm[k] * (spec.lens_bulge * w);
        }
        lens.face_normals();
    }
    out.append(&lens);
    // The inner graphic: light blades across the lamp.
    if !spec.bars.is_empty() {
        let (mut ylo, mut yhi) = (f32::MAX, f32::MIN);
        for p in &lens_src.mesh.pos {
            ylo = ylo.min(p.y);
            yhi = yhi.max(p.y);
        }
        let inner = trim::clip(&lens_src.mesh, &lens_src.field, -0.010);
        for &fb in &spec.bars {
            let yc = ylo + fb * (yhi - ylo);
            let f1: Vec<f32> = inner
                .mesh
                .pos
                .iter()
                .map(|p| p.y - (yc + spec.bar_half))
                .collect();
            let c1 = trim::clip(&inner.mesh, &f1, 0.0);
            if c1.mesh.is_empty() {
                continue;
            }
            let f2: Vec<f32> = c1
                .mesh
                .pos
                .iter()
                .map(|p| (yc - spec.bar_half) - p.y)
                .collect();
            let c2 = trim::clip(&c1.mesh, &f2, 0.0);
            if c2.mesh.is_empty() {
                continue;
            }
            let gm = spec.guides.first().map(|g| g.2).unwrap_or(spec.lens);
            let mut b = trim::offset(&c2.mesh, -0.002, false, Some(gm), 1.0);
            for p in b.pos.iter_mut() {
                *p += spec.axis * (spec.guide_depth + 0.012);
            }
            let loops = trim::boundary_loops(&b);
            for (l, closed) in &loops {
                trim::roll_loop(
                    &mut b,
                    l,
                    *closed,
                    EdgeStyle {
                        gap: 0.0,
                        roll: 0.0015,
                        depth: 0.01,
                        flange: 0.0,
                        ao_floor: 0.8,
                        seal: false,
                    },
                );
            }
            out.append(&b);
        }
    }
    // Housing: the opening's outline swept back along the axis, narrowing.
    let inner = trim::clip(m, f, -0.0005);
    let loops = &inner.loops;
    let centroid = {
        let mut c = Vec3::ZERO;
        for p in &inner.mesh.pos {
            c += *p;
        }
        c / inner.mesh.len().max(1) as f32
    };
    for (l, closed) in loops {
        let mut rings: Vec<Vec<u32>> = Vec::new();
        for (k, t) in [0.0f32, 0.15, 0.5, 1.0].iter().enumerate() {
            let ring = l
                .iter()
                .map(|&v| {
                    let p = inner.mesh.pos[v as usize] - inner.mesh.nrm[v as usize] * 0.004;
                    let shrink = 1.0 - 0.12 * t;
                    let q = centroid + (p - centroid) * shrink + spec.axis * (spec.depth * t);
                    let n = (centroid - p).normalize_or_zero();
                    out.vert(q, n, spec.housing, 0.55 - 0.3 * t + 0.05 * k as f32)
                })
                .collect();
            rings.push(ring);
        }
        if *closed {
            out.strips(&rings, true);
        } else {
            out.strips(&rings, false);
        }
    }
    // Reflector back: the opening's surface pushed back to the housing's end.
    let mut back = trim::offset(&inner.mesh, 0.0, false, Some(spec.reflector), 0.5);
    for p in back.pos.iter_mut() {
        *p = centroid + (*p - centroid) * 0.88 + spec.axis * spec.depth;
    }
    for n in back.nrm.iter_mut() {
        *n = (-spec.axis + *n * 0.4).normalize();
    }
    out.append(&back);
    // Light guides following the outline.
    for &(lo, hi, gm) in &spec.guides {
        let mut g = trim::clip_between(&lens_src.mesh, &lens_src.field, lo, hi);
        if spec.guide_above.len() >= 2 {
            // Height of the line at the nearest of its points (plan view).
            let below: Vec<f32> = g
                .mesh
                .pos
                .iter()
                .map(|p| {
                    let mut best = (f32::MAX, 0.0f32);
                    for w in spec.guide_above.windows(2) {
                        let (a, b) = (Vec2::new(w[0].x, w[0].z), Vec2::new(w[1].x, w[1].z));
                        let q = Vec2::new(p.x, p.z);
                        let ab = b - a;
                        let t = ((q - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                        let d = (a + ab * t).distance(q);
                        if d < best.0 {
                            best = (d, w[0].y + (w[1].y - w[0].y) * t);
                        }
                    }
                    best.1 - p.y
                })
                .collect();
            g = trim::clip(&g.mesh, &below, 0.0);
        }
        let mut gm_mesh = trim::offset(&g.mesh, 0.0, false, Some(gm), 1.0);
        for p in gm_mesh.pos.iter_mut() {
            *p += spec.axis * spec.guide_depth;
        }
        // Give the guide a little thickness (its walls catch the light).
        let loops = trim::boundary_loops(&gm_mesh);
        for (l, closed) in &loops {
            trim::roll_loop(
                &mut gm_mesh,
                l,
                *closed,
                EdgeStyle {
                    gap: 0.0,
                    roll: 0.0015,
                    depth: 0.006,
                    flange: 0.0,
                    ao_floor: 0.8,
                    seal: false,
                },
            );
        }
        out.append(&gm_mesh);
    }
    // Projectors: a chrome cup with a bright lens dome, along the axis.
    for &c in &spec.projectors {
        let r = spec.projector_r;
        let base = c + spec.axis * 0.012;
        let s0 = spec.axis.cross(Vec3::Y).normalize_or(Vec3::X);
        // Cup (axial offsets measured along +axis into the lamp).
        out.lathe(
            base,
            spec.axis,
            s0,
            &[
                (r * 1.25, 0.0),
                (r * 1.1, 0.008),
                (r * 0.9, 0.03),
                (r * 0.5, 0.05),
            ],
            super::segs(32, 10),
            |_, _| mat::CHROME,
            |k| 0.9 - 0.15 * k as f32,
            false,
        );
        // Lens dome (bulging toward the outside, −axis).
        out.lathe(
            base + spec.axis * 0.018,
            spec.axis,
            s0,
            &[
                (r * 0.86, 0.0),
                (r * 0.7, -0.008),
                (r * 0.4, -0.014),
                (0.0, -0.016),
            ],
            super::segs(32, 10),
            |_, _| mat::HEADLIGHT,
            |_| 1.0,
            false,
        );
    }
    out
}

/// A recessed grille in a body opening: surround, throat, back and slats.
#[derive(Clone, Debug)]
pub struct GrilleSpec {
    pub axis: Vec3,
    pub depth: f32,
    /// Surround width (m) and material (chrome / gloss black); 0 for none.
    pub surround: f32,
    pub surround_mat: u8,
    /// Slats: heights (in the field `h` given per vertex) and half thickness;
    /// slat set-back and material.
    pub slats: Vec<(f32, u8)>,
    pub slat_half: f32,
    pub slat_back: f32,
    pub back_mat: u8,
}

pub fn grille(m: &Mesh, f: &[f32], h: &[f32], spec: &GrilleSpec) -> Mesh {
    let mut out = Mesh::default();
    let inner = trim::clip(m, f, 0.0);
    if inner.mesh.is_empty() {
        return out;
    }
    if spec.surround > 0.0 {
        let s = trim::clip_between(m, f, -0.0005, spec.surround);
        let mut sm = trim::offset(&s.mesh, -0.001, false, Some(spec.surround_mat), 1.0);
        let loops = trim::boundary_loops(&sm);
        for (l, closed) in &loops {
            trim::roll_loop(&mut sm, l, *closed, EdgeStyle::TRIM);
        }
        out.append(&sm);
    }
    let centroid = {
        let mut c = Vec3::ZERO;
        for p in &inner.mesh.pos {
            c += *p;
        }
        c / inner.mesh.len().max(1) as f32
    };
    // Throat.
    for (l, closed) in &inner.loops {
        let mut rings = Vec::new();
        for t in [0.0f32, 0.5, 1.0] {
            let ring = l
                .iter()
                .map(|&v| {
                    let p = inner.mesh.pos[v as usize] - inner.mesh.nrm[v as usize] * 0.003;
                    out.vert(
                        p + spec.axis * (spec.depth * t),
                        (centroid - p).normalize_or_zero(),
                        spec.back_mat,
                        0.4 - 0.25 * t,
                    )
                })
                .collect();
            rings.push(ring);
        }
        out.strips(&rings, *closed);
    }
    // Back mesh.
    let mut back = trim::offset(&inner.mesh, 0.0, false, Some(spec.back_mat), 0.25);
    for p in back.pos.iter_mut() {
        *p += spec.axis * spec.depth;
    }
    out.append(&back);
    // Slats: strips of the opening at each height, set back, with walls.
    let hi: Vec<f32> = {
        // h on the clipped opening's vertices: nearest source by position is
        // costly; interpolate by re-clipping a combined field instead.
        Vec::new()
    };
    let _ = hi;
    for &(y, slat_mat) in &spec.slats {
        let band: Vec<f32> = f
            .iter()
            .zip(h)
            .map(|(fv, hv)| fv.max((hv - y).abs() - spec.slat_half))
            .collect();
        let mut c = trim::clip(m, &band, -0.004);
        if c.mesh.is_empty() {
            continue;
        }
        c.mesh.mat.iter_mut().for_each(|x| *x = slat_mat);
        for p in c.mesh.pos.iter_mut() {
            *p += spec.axis * spec.slat_back;
        }
        let loops = std::mem::take(&mut c.loops);
        for (l, closed) in &loops {
            trim::roll_loop(
                &mut c.mesh,
                l,
                *closed,
                EdgeStyle {
                    gap: 0.0,
                    roll: 0.002,
                    depth: 0.02,
                    flange: 0.0,
                    ao_floor: 0.3,
                    seal: false,
                },
            );
        }
        out.append(&c.mesh);
    }
    out
}

/// A wheel's geometry (definition coordinates).
#[derive(Clone, Copy, Debug)]
pub struct WheelSpec {
    pub centre: Vec3,
    /// Outward (the face you see), unit.
    pub out: Vec3,
    pub radius: f32,
    pub width: f32,
    pub rim_radius: f32,
    pub spokes: usize,
    pub disc_radius: f32,
}

/// The turning part of a wheel: tyre, rim, disc.
pub fn wheel(w: &WheelSpec) -> Mesh {
    let mut m = Mesh::default();
    let ax = w.out;
    let s0 = if ax.cross(Vec3::Y).length() > 0.1 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let s0 = (s0 - ax * s0.dot(ax)).normalize();
    let (r, hw, rr) = (w.radius, 0.5 * w.width, w.rim_radius);
    let segs = super::segs(48, 20);
    // Tyre profile from the inner bead round the tread to the outer bead
    // (radius, axial offset).
    let side = |s: f32| -> Vec<(f32, f32)> {
        // One sidewall from the bead (rim) to the shoulder, at axial sign s.
        // Proportional to the sidewall's height (a low-profile tyre on a big
        // rim keeps its bead, rim protector, lettering band and shoulder).
        let sh = r - rr;
        [
            (0.06, 0.86),
            (0.16, 0.93),
            (0.28, 0.99),
            (0.36, 1.025),
            (0.42, 1.035),
            (0.49, 1.02),
            (0.60, 1.03),
            (0.70, 1.045),
            (0.80, 1.035),
            (0.89, 0.99),
            (0.965, 0.90),
            (0.995, 0.80),
        ]
        .iter()
        .map(|&(f, w)| (rr + f * sh, s * hw * w))
        .collect()
    };
    let mut prof: Vec<(f32, f32)> = side(-1.0);
    // Tread with four circumferential grooves.
    let grooves = [-0.55f32, -0.2, 0.2, 0.55];
    let mut tread = Vec::new();
    let n_t = super::segs(14, 6);
    for i in 0..=n_t {
        let u = -0.78 + 1.56 * i as f32 / n_t as f32;
        let mut depth = 0.0;
        for g in grooves {
            if (u - g).abs() < 0.045 {
                depth = 0.008;
            }
        }
        tread.push((r - depth, u * hw));
    }
    prof.extend(tread);
    let mut outer = side(1.0);
    outer.reverse();
    prof.extend(outer);
    let tyre_first = m.len();
    m.lathe(
        w.centre,
        ax,
        s0,
        &prof,
        segs,
        |_, _| mat::TYRE,
        |k| {
            if k < 3 || k > prof.len() - 4 {
                0.6
            } else {
                1.0
            }
        },
        false,
    );
    // The tyre's texture coordinates (lettering and tread blocks in the
    // renderer): u round the wheel (0..1), v the sidewall's height fraction
    // (0 at the bead, 1 at the shoulder; negative on the inner side) or, on
    // the tread, 2 + its width fraction (0..2 across).
    {
        let s1 = ax.cross(s0);
        for k in tyre_first..m.len() {
            let d = m.pos[k] - w.centre;
            let axial = d.dot(ax);
            let radial = (d - ax * axial).length();
            let ang = d.dot(s1).atan2(d.dot(s0));
            let u = (ang / TAU).rem_euclid(1.0);
            let on_tread = radial > r - 0.0095 && axial.abs() < hw * 0.81;
            let v = if on_tread {
                2.0 + (axial / hw + 1.0)
            } else {
                ((radial - rr) / (r - rr)).clamp(0.0, 1.0) * axial.signum()
            };
            m.uv[k] = [u, v];
        }
        // The lettering's seam: a triangle spanning u = 1 → 0 would run the
        // text backwards across it; give its low-u corners copies at u + 1.
        let tyre_tris = m.idx.len();
        let mut copies = std::collections::HashMap::new();
        for t in 0..tyre_tris / 3 {
            let ids = [m.idx[3 * t], m.idx[3 * t + 1], m.idx[3 * t + 2]];
            if ids.iter().any(|&i| (i as usize) < tyre_first) {
                continue;
            }
            let us = ids.map(|i| m.uv[i as usize][0]);
            let (lo, hi) = (
                us.iter().cloned().fold(9.0, f32::min),
                us.iter().cloned().fold(-9.0, f32::max),
            );
            if hi - lo < 0.5 {
                continue;
            }
            for (j, &i) in ids.iter().enumerate() {
                if us[j] < 0.5 {
                    let c = *copies.entry(i).or_insert_with(|| {
                        let i = i as usize;
                        let c = m.vert(m.pos[i], m.nrm[i], m.mat[i], m.ao[i]);
                        m.smooth[c as usize] = m.smooth[i];
                        m.uv[c as usize] = [m.uv[i][0] + 1.0, m.uv[i][1]];
                        c
                    });
                    m.idx[3 * t + j] = c;
                }
            }
        }
    }
    // Rim barrel and lips: outer lip, barrel inside (darker), inner lip.
    let rim_prof = [
        (rr + 0.008, hw * 0.80),
        (rr + 0.006, hw * 0.88),
        (rr - 0.004, hw * 0.90),
        (rr - 0.012, hw * 0.84),
        (rr - 0.016, hw * 0.70),
        (rr - 0.02, hw * 0.4),
        (rr - 0.02, -hw * 0.75),
        (rr - 0.004, -hw * 0.86),
        (rr + 0.006, -hw * 0.86),
    ];
    m.lathe(
        w.centre,
        ax,
        s0,
        &rim_prof,
        segs,
        |k, _| if k < 4 { mat::RIM } else { mat::BRAKE },
        |k| if k < 4 { 1.0 } else { 0.35 },
        true,
    );
    // Spoke face: a dished disc on a polar grid, clipped to the spokes.
    let face_w = |rad: f32| -> f32 {
        // Axial position: hub proud, dished between, rising to the lip.
        let t = ((rad - 0.07) / (rr - 0.085)).clamp(0.0, 1.0);
        hw * 0.80 - 0.035 * (PI * t).sin() * (1.0 - 0.3 * t) + 0.004 * (1.0 - t)
    };
    let nr = super::segs(14, 6);
    let na = super::segs(170, 60);
    let s1 = ax.cross(s0);
    let mut face = Mesh::default();
    let mut ids = vec![vec![0u32; na]; nr + 1];
    let mut field = Vec::new();
    let r_in = 0.0;
    let r_out = rr - 0.008;
    let nsp = w.spokes.max(3);
    for (i, row) in ids.iter_mut().enumerate() {
        let rad = r_in + (r_out - r_in) * (i as f32 / nr as f32).powf(0.85);
        for (j, slot) in row.iter_mut().enumerate() {
            let a = TAU * j as f32 / na as f32;
            let d = s0 * a.cos() + s1 * a.sin();
            let p = w.centre + d * rad + ax * face_w(rad);
            // Normal from the dish slope.
            let e = 0.002;
            let dw = (face_w(rad + e) - face_w((rad - e).max(0.0))) / (2.0 * e);
            let n = (ax - d * dw).normalize();
            *slot = face.vert(p, n, mat::RIM, 1.0);
            // Split spokes: each spoke forks into two thin arms toward the
            // lip (a Y), the arms' faces diamond cut.
            let sector = TAU / nsp as f32;
            let k = (a / sector).round();
            let da = a - k * sector;
            let split = 0.062 * smoothstep(0.11, rr - 0.03, rad);
            let arm = (da.abs() - split).abs() * rad;
            let half = 0.0105
                + 0.007 * (1.0 - smoothstep(0.07, 0.15, rad))
                + 0.006 * smoothstep(rr - 0.05, rr - 0.012, rad);
            let spoke = arm - half;
            let hub = rad - 0.078;
            let lip = (rr - 0.020) - rad;
            field.push(spoke.min(hub).min(lip));
        }
    }
    for i in 0..nr {
        for j in 0..na {
            let j1 = (j + 1) % na;
            face.quad(ids[i][j], ids[i][j1], ids[i + 1][j1], ids[i + 1][j]);
        }
    }
    face.orient(|_| ax);
    let mut c = trim::clip(&face, &field, -0.0015);
    // The faces are machined; the walls the roll adds stay gunmetal.
    c.mesh.mat.iter_mut().for_each(|x| *x = mat::MACHINED);
    let face_n = c.mesh.len();
    let loops = std::mem::take(&mut c.loops);
    for (l, closed) in &loops {
        trim::roll_loop(
            &mut c.mesh,
            l,
            *closed,
            EdgeStyle {
                gap: 0.0,
                roll: 0.003,
                depth: 0.04,
                flange: 0.0,
                ao_floor: 0.35,
                seal: false,
            },
        );
    }
    for k in face_n..c.mesh.len() {
        c.mesh.mat[k] = mat::RIM;
    }
    m.append(&c.mesh);
    // Centre cap and lug nuts.
    let hub_w = face_w(0.0);
    m.lathe(
        w.centre + ax * hub_w,
        ax,
        s0,
        &[(0.032, 0.0), (0.03, 0.006), (0.02, 0.01), (0.0, 0.011)],
        32,
        |_, _| mat::CHROME,
        |_| 1.0,
        false,
    );
    for k in 0..5 {
        let a = TAU * (k as f32 + 0.5) / 5.0;
        let d = s0 * a.cos() + s1 * a.sin();
        let c = w.centre + d * 0.056 + ax * (hub_w - 0.002);
        m.lathe(
            c,
            ax,
            s0,
            &[
                (0.0105, 0.0),
                (0.0105, 0.009),
                (0.008, 0.013),
                (0.0, 0.0135),
            ],
            12,
            |_, _| mat::CHROME,
            |_| 0.85,
            false,
        );
    }
    // Brake disc (vented, with its hat), behind the spokes.
    let dw = hw * 0.15;
    let disc = [
        (w.disc_radius * 0.52, dw + 0.013),
        (w.disc_radius, dw + 0.013),
        (w.disc_radius, dw - 0.013),
        (w.disc_radius * 0.52, dw - 0.013),
    ];
    m.lathe(
        w.centre,
        ax,
        s0,
        &disc,
        super::segs(48, 16),
        |_, _| mat::BRAKE,
        |k| if k == 0 { 0.8 } else { 0.6 },
        true,
    );
    m.lathe(
        w.centre,
        ax,
        s0,
        &[
            (w.disc_radius * 0.52, dw + 0.013),
            (0.07, hw * 0.55),
            (0.0, hw * 0.55),
        ],
        48,
        |_, _| mat::BRAKE,
        |_| 0.5,
        false,
    );
    m
}

/// The brake caliper (does not turn): a block hugging the disc at `angle`
/// (radians from up, toward +z… measured in the wheel's plane from s0 = up).
pub fn caliper(w: &WheelSpec, angle_from_up: f32, toward: Vec3) -> Mesh {
    let mut m = Mesh::default();
    let ax = w.out;
    let up = (Vec3::Y - ax * Vec3::Y.dot(ax)).normalize();
    let fwd = toward.normalize();
    let dw = 0.5 * w.width * 0.15; // the disc's mid plane (as in `wheel`)
    let rd = w.disc_radius;
    // A monoblock caliper: an outboard piston housing with a flat, crowned
    // face, a bridge over the disc's rim, and a slimmer inboard half. Each is
    // a convex rounded section (axial a, radial ρ) swept round the disc's
    // edge; the ends are rounded off in plan.
    let half = 0.40; // half the arc (rad): about 15 cm long at the front
    let n = 14;
    let block = |m: &mut Mesh, a0: f32, a1: f32, r0: f32, r1: f32, rc: f32, sweep: f32| {
        let sec = rounded_rect(a0, a1, r0, r1, rc, 4);
        let rings: Vec<Vec<Vec3>> = (0..=n)
            .map(|k| {
                let t = k as f32 / n as f32 * 2.0 - 1.0;
                // Rounded ends: the section shrinks toward its centre.
                let s = (1.0 - t.abs().powi(6)).max(0.0).powf(0.35).max(0.55);
                let ang = angle_from_up + t * sweep;
                let (ca, sa) = (ang.cos(), ang.sin());
                let rad = up * ca + fwd * sa;
                let (am, rm) = (0.5 * (a0 + a1), 0.5 * (r0 + r1));
                sec.iter()
                    .map(|&[a, r]| {
                        let a = am + (a - am) * s;
                        let r = rm + (r - rm) * s;
                        w.centre + rad * r + ax * (dw + a)
                    })
                    .collect()
            })
            .collect();
        m.loft(&rings, |_| mat::CALIPER, 0.85, true);
    };
    // Outboard housing: the face toward the spokes.
    block(&mut m, 0.016, 0.042, rd - 0.058, rd + 0.016, 0.009, half);
    // Bridge over the disc's edge.
    block(
        &mut m,
        -0.036,
        0.036,
        rd + 0.004,
        rd + 0.026,
        0.008,
        half * 0.82,
    );
    // Inboard half.
    block(
        &mut m,
        -0.040,
        -0.016,
        rd - 0.046,
        rd + 0.016,
        0.008,
        half * 0.9,
    );
    m
}

/// A rounded rectangle [a0, a1] × [r0, r1] (corner radius `rc`, `k` segments a
/// corner), counter-clockwise.
fn rounded_rect(a0: f32, a1: f32, r0: f32, r1: f32, rc: f32, k: usize) -> Vec<[f32; 2]> {
    let corners = [
        (a1 - rc, r0 + rc, -0.5 * PI),
        (a1 - rc, r1 - rc, 0.0),
        (a0 + rc, r1 - rc, 0.5 * PI),
        (a0 + rc, r0 + rc, PI),
    ];
    let mut out = Vec::new();
    for (cx, cy, start) in corners {
        for i in 0..=k {
            let t = start + 0.5 * PI * i as f32 / k as f32;
            out.push([cx + rc * t.cos(), cy + rc * t.sin()]);
        }
    }
    out
}

/// A mirror: housing (an egg, cut flat at the back for the glass) on an arm
/// from the door's sail. `base` on the body, `out` outward (unit), `back`
/// rearward (unit).
pub fn mirror(base: Vec3, out: Vec3, back: Vec3, paint: u8) -> Mesh {
    let mut m = Mesh::default();
    let up = Vec3::Y;
    // Housing centre: out and up from the base.
    let c = base + out * 0.12 + up * 0.035 + back * 0.02;
    let (lx, ly, lz) = (0.115, 0.07, 0.055); // half sizes: outward, up, fore-aft
    let n_u = 24;
    let n_v = 16;
    let mut rings = Vec::new();
    for i in 0..=n_v {
        // Fore-aft from the nose (−back) to the glass face (+back).
        let t = i as f32 / n_v as f32;
        let s = (PI * 0.5 * t.min(0.999)).sin().powf(0.5);
        let s = if t > 0.92 {
            s * (1.0 - (t - 0.92) * 3.0)
        } else {
            s
        };
        let zz = -lz + 2.0 * lz * (1.0 - (1.0 - t).powf(1.6));
        let mut ring = Vec::new();
        for j in 0..n_u {
            let a = TAU * j as f32 / n_u as f32;
            let (ca, sa) = (a.cos(), a.sin());
            let sx = ca.signum() * ca.abs().powf(0.7);
            let sy = sa.signum() * sa.abs().powf(0.7);
            let p = c + out * (sx * lx * s) + up * (sy * ly * s) + back * zz;
            let n = (out * (ca / lx) + up * (sa / ly) + back * ((t - 0.5) * 6.0)).normalize();
            let mtl = if sa < -0.35 { mat::TRIM } else { paint };
            ring.push(m.vert(p, n, mtl, 1.0));
        }
        rings.push(ring);
    }
    m.strips(&rings, true);
    // The glass (flat, set into the back face).
    let gc = c + back * (lz * 0.98);
    let gid = m.vert(gc, back, mat::CHROME, 0.9);
    let last = rings.last().unwrap().clone();
    let mut inner = Vec::new();
    for &v in &last {
        let p = m.pos[v as usize];
        let q = gc + (p - gc) * 0.9 - back * 0.006;
        inner.push(m.vert(q, back, mat::CHROME, 0.9));
    }
    m.strips(&[last.clone(), inner.clone()], true);
    for i in 0..inner.len() {
        m.tri(gid, inner[i], inner[(i + 1) % inner.len()]);
    }
    // Arm: from the base to the housing's inner lower face.
    let a0 = base - out * 0.01;
    let a1 = c - out * (lx * 0.6) - up * (ly * 0.2);
    let path: Vec<Vec3> = (0..=6)
        .map(|k| {
            let t = k as f32 / 6.0;
            a0.lerp(a1, t) + up * (0.012 * (PI * t).sin())
        })
        .collect();
    m.tube(&path, up, 0.022, 0.012, 12, mat::TRIM, 0.8);
    m
}

/// A pull handle: a chrome bar over a dark pocket, lying on the surface
/// points `path` (front to rear) with outward normals `nrm`.
pub fn handle(path: &[Vec3], nrm: &[Vec3], bar: u8) -> Mesh {
    let mut m = Mesh::default();
    let n = path.len();
    // Pocket: a dark lozenge just proud of the skin.
    let mut top = Vec::new();
    let mut bot = Vec::new();
    for k in 0..n {
        let t = k as f32 / (n - 1) as f32;
        let w = 0.022 * (PI * t).sin().powf(0.4);
        let p = path[k] + nrm[k] * 0.0012;
        top.push(m.vert(p + Vec3::Y * (w + 0.004), nrm[k], mat::TRIM, 0.5));
        bot.push(m.vert(p - Vec3::Y * (w + 0.006), nrm[k], mat::TRIM, 0.35));
    }
    m.strips(&[top, bot], false);
    // Bar: a flattened tube standing off the pocket.
    let bar_path: Vec<Vec3> = (1..n - 1)
        .map(|k| {
            let t = k as f32 / (n - 1) as f32;
            path[k] + nrm[k] * (0.006 + 0.006 * (PI * t).sin().powf(0.3))
        })
        .collect();
    let up = Vec3::Y;
    m.tube(&bar_path, up, 0.006, 0.012, 14, bar, 1.0);
    m
}

/// A wheel-arch liner: a dark half tube inside the arch.
pub fn arch_liner(centre: Vec3, side: f32, radius: f32, x_out: f32, x_in: f32) -> Mesh {
    let mut m = Mesh::default();
    let mut rings = Vec::new();
    for k in 0..=2 {
        let x = side * (x_out + (x_in - x_out) * k as f32 / 2.0);
        let mut ring = Vec::new();
        for i in 0..=48 {
            let a = -0.35 + (PI + 0.7) * i as f32 / 48.0;
            let d = Vec2::new(a.cos(), a.sin());
            let p = Vec3::new(x, centre.y + d.y * radius, centre.z + d.x * radius);
            ring.push(m.vert(
                p,
                Vec3::new(0.0, -d.y, -d.x),
                mat::UNDERBODY,
                0.35 - 0.1 * k as f32,
            ));
        }
        rings.push(ring);
    }
    m.strips(&rings, false);
    m
}

/// The underbody: a dark plate between the band's bottom edges.
pub fn underbody(rows: &[(f32, f32, f32)]) -> Mesh {
    // rows: (z, y, half width)
    let mut m = Mesh::default();
    let mut rings = Vec::new();
    for &(z, y, w) in rows {
        let ring: Vec<u32> = (0..=8)
            .map(|k| {
                let x = -w + 2.0 * w * k as f32 / 8.0;
                m.vert(Vec3::new(x, y, z), -Vec3::Y, mat::UNDERBODY, 0.5)
            })
            .collect();
        rings.push(ring);
    }
    m.strips(&rings, false);
    m
}

/// An exhaust: a pipe from under the floor to a chrome tip at `tip` (pointing
/// along `dir`).
pub fn exhaust(from: Vec3, tip: Vec3, dir: Vec3, r: f32) -> Mesh {
    let mut m = Mesh::default();
    let mid = tip - dir * 0.15;
    let path: Vec<Vec3> = (0..=8)
        .map(|k| {
            let t = k as f32 / 8.0;
            from.lerp(mid, t)
        })
        .collect();
    m.tube(&path, Vec3::Y, r * 0.6, r * 0.6, 12, mat::UNDERBODY, 0.4);
    let s0 = Vec3::Y;
    m.lathe(
        tip,
        dir,
        s0,
        &[
            (r * 0.85, -0.16),
            (r, -0.07),
            (r, 0.0),
            (r * 0.97, 0.004),
            (r * 0.85, 0.0),
            (r * 0.85, -0.05),
        ],
        32,
        |k, _| if k >= 4 { mat::TRIM } else { mat::CHROME },
        |k| if k >= 4 { 0.15 } else { 1.0 },
        false,
    );
    m
}

/// A flat polygon (convex, counter-clockwise in (right, up)) extruded `depth`
/// out of the surface along `n`, with walls: a badge's facet.
fn facet(
    m: &mut Mesh,
    c: Vec3,
    n: Vec3,
    right: Vec3,
    up: Vec3,
    pts: &[[f32; 2]],
    depth: f32,
    mtl: u8,
) {
    let at = |p: [f32; 2], d: f32| c + right * p[0] + up * p[1] + n * d;
    let top: Vec<u32> = pts
        .iter()
        .map(|&p| m.vert(at(p, depth), n, mtl, 1.0))
        .collect();
    for k in 1..pts.len() - 1 {
        m.tri(top[0], top[k], top[k + 1]);
    }
    for k in 0..pts.len() {
        let (a, b) = (pts[k], pts[(k + 1) % pts.len()]);
        let e = (at(b, 0.0) - at(a, 0.0)).normalize_or_zero();
        let o = e.cross(n).normalize_or_zero();
        let q = [
            m.vert(at(a, 0.0), o, mtl, 0.7),
            m.vert(at(b, 0.0), o, mtl, 0.7),
            m.vert(at(b, depth), o, mtl, 1.0),
            m.vert(at(a, depth), o, mtl, 1.0),
        ];
        m.quad(q[0], q[1], q[2], q[3]);
    }
}

/// The car's badge, an original mark: a hexagonal ring split by a diagonal
/// into two shards (rubble), in chrome on a black backing. `c` on the surface,
/// facing `n`; `size` its width (m).
pub fn emblem(c: Vec3, n: Vec3, up: Vec3, size: f32) -> Mesh {
    let mut m = Mesh::default();
    let n = n.normalize();
    let right = up.cross(n).normalize();
    let up = n.cross(right).normalize();
    let r = 0.5 * size;
    let hex = |rad: f32, k: usize| -> [f32; 2] {
        let a = PI / 6.0 + TAU * k as f32 / 6.0;
        [rad * a.cos(), rad * a.sin() * 0.82]
    };
    // Backing.
    let back: Vec<[f32; 2]> = (0..6).map(|k| hex(r * 1.06, k)).collect();
    facet(&mut m, c, n, right, up, &back, 0.002, mat::TRIM);
    // The ring: six trapezoids.
    for k in 0..6 {
        let quad = [
            hex(r * 0.8, k),
            hex(r * 0.8, k + 1),
            hex(r, k + 1),
            hex(r, k),
        ];
        let ccw = [quad[0], quad[3], quad[2], quad[1]];
        let pick = if (quad[1][0] - quad[0][0]) * (quad[2][1] - quad[0][1])
            - (quad[1][1] - quad[0][1]) * (quad[2][0] - quad[0][0])
            > 0.0
        {
            quad
        } else {
            ccw
        };
        facet(&mut m, c, n, right, up, &pick, 0.005, mat::CHROME);
    }
    // Two shards split by a diagonal gap.
    let g = 0.06 * r;
    let s1 = [
        [-0.55 * r, -0.42 * r],
        [0.42 * r - g, -0.42 * r],
        [-0.55 * r, 0.4 * r - g],
    ];
    let s2 = [
        [0.55 * r, 0.42 * r],
        [-0.42 * r + g, 0.42 * r],
        [0.55 * r, -0.4 * r + g],
    ];
    facet(&mut m, c, n, right, up, &s1, 0.006, mat::CHROME);
    facet(&mut m, c, n, right, up, &s2, 0.006, mat::CHROME);
    m
}
