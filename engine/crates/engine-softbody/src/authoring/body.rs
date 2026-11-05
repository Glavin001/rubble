//! The body's shape and its outer shell of nodes.
//!
//! A [`BodyProfile`] describes the car the way a designer's package drawing does:
//! curves along the length (z from the rear end, 0, to the front, `length`) for the
//! roof/bonnet/deck centre line, the underbody, the shoulder (where the side meets
//! the top surface), the side's half-width and the top surface's half-width
//! (tumblehome), plus wheel wells and the nose and tail domes. [`BodyProfile::half_section`]
//! turns them into a cross-section at any z, a closed loop (right half from the
//! bottom centre round to the top centre; the left half mirrors it).
//!
//! [`build_shell`] lofts rings of nodes along the length (spacing per surface
//! zone), caps both ends with domed rows of nodes, triangulates the lot into a
//! closed skin, and lists the beams that make it a braced shell: every triangle
//! edge, the opposite diagonal of every quad (X bracing) and bend beams that skip
//! a node along and across the rings.

use super::curve::Curve;
use super::skeleton::Skeleton;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Which surface of the section a point lies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Seg {
    /// Underbody.
    Floor,
    /// Wheel-well wall and ceiling.
    Well,
    /// The lower corner (sill / rocker).
    Lower,
    /// The side between sill and shoulder.
    Side,
    /// Above the shoulder: greenhouse side (windows, pillars) or fender top.
    Upper,
    /// The top surface: roof, screens, bonnet, deck.
    Top,
    /// A domed end cap (nose or tail).
    Cap,
}

/// A wheel well: arch centre (z along the car, y height), radius, and the x of its
/// inner wall.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Well {
    pub z: f32,
    pub y: f32,
    pub radius: f32,
    pub inner_x: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyProfile {
    pub length: f32,
    /// Centre-line height of the top surface (roof, screens, bonnet, deck).
    pub top: Curve,
    /// Underbody height.
    pub bottom: Curve,
    /// Shoulder height: where the side meets the greenhouse or bonnet.
    pub shoulder: Curve,
    /// Half-width of the side.
    pub half_width: Curve,
    /// Half-width of the top surface's edge (narrower over the cabin: tumblehome).
    pub top_half_width: Curve,
    /// The top surface's camber at the centre (m).
    pub crown: f32,
    /// The side's outward bulge between sill and shoulder (m).
    pub side_bulge: f32,
    pub lower_radius: f32,
    pub top_radius: f32,
    /// Depth of the nose and tail domes (m): the end caps bulge this far.
    pub front_sweep: f32,
    pub rear_sweep: f32,
    pub wells: Vec<Well>,
}

/// One point of a section: position (x ≥ 0, y) and surface.
#[derive(Clone, Copy, Debug)]
pub struct SectionPoint {
    pub p: Vec2,
    pub seg: Seg,
}

impl BodyProfile {
    /// Height of the wheel-well ceiling at z (the underbody where there is none).
    pub fn arch(&self, z: f32) -> Option<(f32, f32)> {
        let mut best: Option<(f32, f32)> = None;
        for w in &self.wells {
            let dz = z - w.z;
            if dz.abs() < w.radius {
                let y = w.y + (w.radius * w.radius - dz * dz).sqrt();
                if best.is_none_or(|b| y > b.0) {
                    best = Some((y, w.inner_x));
                }
            }
        }
        best
    }

    /// The right half (x ≥ 0) of the section at z as a dense polyline from the
    /// bottom centre to the top centre, each point with its surface.
    pub fn half_section(&self, z: f32) -> Vec<SectionPoint> {
        let yb = self.bottom.eval(z);
        let ys = self.shoulder.eval(z);
        let yt = self.top.eval(z);
        let hw = self.half_width.eval(z);
        let lr = self.lower_radius;
        let tr = self.top_radius;
        let thw = self.top_half_width.eval(z).min(hw - 0.02).max(tr + 0.05);
        let (ya, xw) = match self.arch(z) {
            Some((y, x)) => (y.max(yb), x),
            None => (yb, self.wells.first().map_or(hw - 2.0 * lr, |w| w.inner_x)),
        };
        let xw = xw.min(hw - lr - 0.02);
        let mut out = Vec::new();
        let mut line = |out: &mut Vec<SectionPoint>, a: Vec2, b: Vec2, seg: Seg, n: usize| {
            for i in 0..n {
                let t = i as f32 / n as f32;
                out.push(SectionPoint {
                    p: a.lerp(b, t),
                    seg,
                });
            }
        };
        let p0 = Vec2::new(0.0, yb);
        let p1 = Vec2::new(xw, yb);
        let p2 = Vec2::new(xw, ya);
        let p3 = Vec2::new(hw - lr, ya);
        line(&mut out, p0, p1, Seg::Floor, 40);
        if ya > yb + 1e-3 {
            line(&mut out, p1, p2, Seg::Well, 20);
            line(&mut out, p2, p3, Seg::Well, 12);
        } else {
            line(&mut out, p1, p3, Seg::Floor, 12);
        }
        // Lower corner: a quarter circle about (hw − lr, ya + lr).
        let side_lo = (ya + lr).min(ys - 0.05);
        let c = Vec2::new(hw - lr, side_lo);
        for i in 0..10 {
            let a = -std::f32::consts::FRAC_PI_2 + i as f32 / 10.0 * std::f32::consts::FRAC_PI_2;
            let r = Vec2::new(lr * a.cos(), (side_lo - ya) * a.sin());
            out.push(SectionPoint {
                p: c + r,
                seg: Seg::Lower,
            });
        }
        // Side: bulging between the corner and the shoulder.
        for i in 0..24 {
            let t = i as f32 / 24.0;
            let y = side_lo + (ys - side_lo) * t;
            let x = hw + self.side_bulge * (std::f32::consts::PI * t).sin();
            out.push(SectionPoint {
                p: Vec2::new(x, y),
                seg: Seg::Side,
            });
        }
        // Upper: shoulder to the top surface's edge.
        let ps = Vec2::new(hw, ys);
        let pe = Vec2::new(thw, (yt - tr).max(ys + 0.005));
        line(&mut out, ps, pe, Seg::Upper, 24);
        // Top edge: a quarter circle to the top surface, then the cambered top.
        let ce = Vec2::new(thw - tr, pe.y);
        let rise = (yt - pe.y).max(0.005);
        for i in 0..10 {
            let a = i as f32 / 10.0 * std::f32::consts::FRAC_PI_2;
            out.push(SectionPoint {
                p: ce + Vec2::new(tr * a.cos(), rise * a.sin()),
                seg: Seg::Top,
            });
        }
        let xe = thw - tr;
        for i in 0..=30 {
            let x = xe * (1.0 - i as f32 / 30.0);
            let u = x / xe;
            out.push(SectionPoint {
                p: Vec2::new(x, yt + self.crown * (1.0 - u * u)),
                seg: Seg::Top,
            });
        }
        out
    }

    /// The length the loft runs over (between the domes).
    pub fn loft_range(&self) -> (f32, f32) {
        (self.rear_sweep, self.length - self.front_sweep)
    }
}

/// Node spacing (m) per surface.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spacing {
    /// Between rings (along the length).
    pub ring: f32,
    pub side: f32,
    pub top: f32,
    pub floor: f32,
    /// Between nodes of a cap's rows.
    pub cap: f32,
}

impl Spacing {
    /// The one-tier shell (BeamNG's density, the BX's 0.19–0.25 m): every
    /// spacing a pole can meet (along the car, the sides and the ends) at
    /// most [`super::vehicle::POLE_LANE`], so a 254 mm pole always meets a
    /// node of the shell. The roof and the floor, which no pole meets,
    /// keep the resolution's coarser spacing across the car (Medium 0.30
    /// and 0.35 m): fewer, heavier nodes (the BX body's median node is
    /// 1.7 kg).
    pub fn pole_lanes(self) -> Self {
        let p = super::vehicle::POLE_LANE;
        Self {
            ring: self.ring.min(p),
            side: self.side.min(p),
            top: self.top,
            floor: self.floor,
            cap: self.cap.min(p),
        }
    }

    pub fn of(&self, seg: Seg) -> f32 {
        match seg {
            Seg::Floor | Seg::Well => self.floor,
            Seg::Lower | Seg::Side | Seg::Upper => self.side,
            Seg::Top => self.top,
            Seg::Cap => self.cap,
        }
    }
}

/// Points along a polyline at even weighted arc length (weight 1/spacing of each
/// point's surface): `n` segments, so n + 1 points, first and last included.
pub fn place(poly: &[SectionPoint], spacing: &Spacing) -> Vec<SectionPoint> {
    let mut acc = vec![0.0f32];
    for w in poly.windows(2) {
        let l = w[0].p.distance(w[1].p);
        let wgt = 1.0 / spacing.of(w[0].seg);
        acc.push(acc.last().unwrap() + l * wgt);
    }
    let total = *acc.last().unwrap();
    let n = total.round().max(2.0) as usize;
    let mut out = Vec::with_capacity(n + 1);
    let mut j = 0;
    for i in 0..=n {
        let target = total * i as f32 / n as f32;
        while j + 1 < acc.len() - 1 && acc[j + 1] < target {
            j += 1;
        }
        let span = (acc[j + 1] - acc[j]).max(1e-9);
        let t = ((target - acc[j]) / span).clamp(0.0, 1.0);
        let p = poly[j].p.lerp(poly[j + 1].p, t);
        let seg = if t < 0.5 {
            poly[j].seg
        } else {
            poly[j + 1].seg
        };
        out.push(SectionPoint { p, seg });
    }
    out
}

/// The ring stations: `keys` kept (merged when closer than [`KEY_MERGE`]), the
/// gaps between split evenly at most `spacing` × 1.1 apart.
pub fn stations(z0: f32, z1: f32, keys: &[f32], spacing: f32) -> Vec<f32> {
    let mut k: Vec<f32> = keys
        .iter()
        .copied()
        .filter(|&z| z > z0 + KEY_MERGE && z < z1 - KEY_MERGE)
        .collect();
    k.push(z0);
    k.push(z1);
    k.sort_by(f32::total_cmp);
    let mut merged: Vec<f32> = Vec::new();
    for z in k {
        if let Some(&last) = merged.last() {
            if z - last < KEY_MERGE {
                continue;
            }
        }
        merged.push(z);
    }
    if *merged.last().unwrap() < z1 - 1e-4 {
        *merged.last_mut().unwrap() = z1;
    }
    let mut out = vec![merged[0]];
    for w in merged.windows(2) {
        let n = ((w[1] - w[0]) / (spacing * 1.1)).ceil().max(1.0) as usize;
        for i in 1..=n {
            out.push(w[0] + (w[1] - w[0]) * i as f32 / n as f32);
        }
    }
    out
}

/// Ring stations closer than this (m) are one: keys are deliberate (door
/// seams, pillars), so they are kept however sparse the rings between them.
pub const KEY_MERGE: f32 = 0.05;

/// A ring of the shell: its station and its right half's nodes from the bottom
/// centre (first) to the top centre (last), with their weighted arc fraction.
#[derive(Clone, Debug)]
pub struct ShellRing {
    pub z: f32,
    pub half: Vec<u32>,
    pub v: Vec<f32>,
}

/// The shell's output (nodes are added to the builder by the caller's closure).
#[derive(Clone, Debug, Default)]
pub struct Shell {
    pub rings: Vec<ShellRing>,
    /// Surface of every shell node (by node index; others absent).
    pub seg: std::collections::HashMap<u32, Seg>,
    /// Mirror of every shell node (centre-line nodes map to themselves).
    pub mirror: std::collections::HashMap<u32, u32>,
    /// Cap rows, bottom to top, each left to right (x increasing); rear then front.
    pub caps: [Vec<Vec<u32>>; 2],
    /// Skin triangles (outward) and beams: (a, b, kind).
    pub tris: Vec<[u32; 3]>,
    pub beams: Vec<(u32, u32, ShellBeam)>,
    pub nodes: Vec<u32>,
    /// The caps' triangles (unoriented).
    pub cap_tris: Vec<[u32; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellBeam {
    /// A triangle edge.
    Edge,
    /// The other diagonal of a quad.
    Brace,
    /// Skips a node (bending stiffness).
    Bend,
}

/// Zipper two ordered node lists by their parameters into triangles (a's side
/// first in each triangle's winding: (a_i, a_i+1, b_j) or (a_i, b_j+1, b_j)).
fn zipper(a: &[u32], va: &[f32], b: &[u32], vb: &[f32], out: &mut Vec<[u32; 3]>) {
    let (mut i, mut j) = (0usize, 0usize);
    while i + 1 < a.len() || j + 1 < b.len() {
        let adv_a = if i + 1 >= a.len() {
            false
        } else if j + 1 >= b.len() {
            true
        } else {
            // Advance the side whose next point is nearer in parameter.
            va[i + 1] - vb[j] < vb[j + 1] - va[i]
        };
        if adv_a {
            out.push([a[i], a[i + 1], b[j]]);
            i += 1;
        } else {
            out.push([a[i], b[j + 1], b[j]]);
            j += 1;
        }
    }
}

/// Build the shell into `sk` (every node in part 0). `keys`: ring stations to
/// keep (seams, pillars, axles).
pub fn build_shell(
    profile: &BodyProfile,
    spacing: &Spacing,
    keys: &[f32],
    sk: &mut Skeleton,
) -> Shell {
    let (z0, z1) = profile.loft_range();
    let zs = stations(z0, z1, keys, spacing.ring);
    let mut shell = Shell::default();
    let add = |sk: &mut Skeleton, shell: &mut Shell, p: Vec3, seg: Seg| -> u32 {
        let i = sk.node(p, 0);
        shell.seg.insert(i, seg);
        shell.nodes.push(i);
        i
    };
    // Rings: right half placed, the left half mirrored (centre-line nodes shared).
    for &z in &zs {
        let pts = place(&profile.half_section(z), spacing);
        let n = pts.len() - 1;
        let mut half = Vec::with_capacity(n + 1);
        let mut v = Vec::with_capacity(n + 1);
        for (i, sp) in pts.iter().enumerate() {
            let x = if i == 0 || i == n { 0.0 } else { sp.p.x };
            let r = add(sk, &mut shell, Vec3::new(x, sp.p.y, z), sp.seg);
            if i == 0 || i == n {
                shell.mirror.insert(r, r);
            } else {
                let l = add(sk, &mut shell, Vec3::new(-x, sp.p.y, z), sp.seg);
                shell.mirror.insert(r, l);
                shell.mirror.insert(l, r);
            }
            half.push(r);
            v.push(i as f32 / n as f32);
        }
        shell.rings.push(ShellRing { z, half, v });
    }
    let mut loft: Vec<[u32; 3]> = Vec::new();
    for k in 0..shell.rings.len() - 1 {
        let (a, b) = (&shell.rings[k], &shell.rings[k + 1]);
        zipper(&a.half, &a.v, &b.half, &b.v, &mut loft);
    }
    let mut all: Vec<[u32; 3]> = Vec::new();
    for t in &loft {
        all.push(*t);
        all.push([
            shell.mirror[&t[0]],
            shell.mirror[&t[2]],
            shell.mirror[&t[1]],
        ]);
    }
    // Caps: rows across the end rings, domed outward. Ring nodes closer in height
    // than 0.6 × the cap spacing share a row.
    let merge = 0.6 * spacing.cap;
    for (ci, front) in [(0usize, false), (1, true)] {
        let ring = if front {
            shell.rings.last().unwrap().clone()
        } else {
            shell.rings[0].clone()
        };
        let sweep = if front {
            profile.front_sweep
        } else {
            -profile.rear_sweep
        };
        let segs: Vec<Seg> = ring.half.iter().map(|i| shell.seg[i]).collect();
        let first_side = segs
            .iter()
            .position(|s| !matches!(s, Seg::Floor | Seg::Well))
            .unwrap_or(1);
        let first_top = segs
            .iter()
            .position(|s| *s == Seg::Top)
            .unwrap_or(ring.half.len() - 1);
        let yb = sk.pos(ring.half[0]).y;
        let y_top_edge = sk.pos(ring.half[first_top]).y;
        let yt = sk.pos(*ring.half.last().unwrap()).y;
        // Groups of ring nodes: the floor's, each side row's, the top's.
        let mut groups: Vec<Vec<u32>> = vec![ring.half[..first_side].to_vec()];
        let mut side: Vec<Vec<u32>> = Vec::new();
        let mut top_extra: Vec<u32> = Vec::new();
        for j in first_side..first_top {
            let r = ring.half[j];
            let y = sk.pos(r).y;
            if y - yb < merge {
                groups[0].push(r);
            } else if y_top_edge - y < merge {
                top_extra.push(r);
            } else if let Some(g) = side.last_mut().filter(|g| y - sk.pos(g[0]).y < merge) {
                g.push(r);
            } else {
                side.push(vec![r]);
            }
        }
        let mut top: Vec<u32> = top_extra;
        top.extend_from_slice(&ring.half[first_top..]);
        groups.extend(side);
        groups.push(top);
        let last = groups.len() - 1;
        // Each row twice: as the lower row of a strip (its ring nodes in loop order
        // at both ends) and as the upper one (only its lowest ring node at each end),
        // so the strips' boundary runs through every ring node in loop order.
        let mut rows: Vec<Vec<u32>> = Vec::new();
        let mut lower_rows: Vec<(Vec<u32>, Vec<f32>)> = Vec::new();
        let mut upper_rows: Vec<(Vec<u32>, Vec<f32>)> = Vec::new();
        for (gi, g) in groups.iter().enumerate() {
            let mirrored = |i: u32| shell.mirror[&i];
            if gi == 0 || gi == last {
                // Floor and top rows: their ring nodes ordered by x from the left.
                let mut row: Vec<u32> = Vec::new();
                for &i in g {
                    row.push(i);
                    if mirrored(i) != i {
                        row.push(mirrored(i));
                    }
                }
                if gi == 0 {
                    // Loop order: left half reversed, then the right half.
                    let right: Vec<u32> = g.to_vec();
                    let mut ordered: Vec<u32> = right
                        .iter()
                        .rev()
                        .filter(|&&i| mirrored(i) != i)
                        .map(|&i| mirrored(i))
                        .collect();
                    ordered.extend(right.iter().copied());
                    row = ordered;
                } else {
                    row.sort_by(|&p, &q| sk.pos(p).x.total_cmp(&sk.pos(q).x).then(p.cmp(&q)));
                }
                let par: Vec<f32> = row.iter().map(|&i| sk.pos(i).x).collect();
                // Monotone parameters for the loop-ordered floor row.
                let par = monotone(&par);
                rows.push(row.clone());
                lower_rows.push((row.clone(), par.clone()));
                upper_rows.push((row, par));
                continue;
            }
            // A side row: interior nodes between the lowest ring node and its mirror.
            let g0 = g[0];
            let xr = sk.pos(g0).x;
            let y = sk.pos(g0).y;
            let m = ((2.0 * xr) / spacing.cap).round().max(1.0) as usize;
            let taper = (std::f32::consts::PI * ((y - yb) / (yt - yb).max(1e-3)).clamp(0.0, 1.0))
                .sin()
                .sqrt();
            let mut interior = Vec::new();
            for i in 1..m {
                let x = -xr + 2.0 * xr * i as f32 / m as f32;
                let u = x / xr;
                let dz = sweep * (1.0 - u * u) * taper;
                interior.push(add(sk, &mut shell, Vec3::new(x, y, ring.z + dz), Seg::Cap));
            }
            let len = interior.len();
            for i in 0..len {
                shell.mirror.insert(interior[i], interior[len - 1 - i]);
            }
            let mut upper = vec![shell.mirror[&g0]];
            upper.extend(interior.iter().copied());
            upper.push(g0);
            let mut lower: Vec<u32> = g.iter().rev().map(|&i| shell.mirror[&i]).collect();
            lower.extend(interior.iter().copied());
            lower.extend(g.iter().copied());
            let upar: Vec<f32> = upper.iter().map(|&i| sk.pos(i).x).collect();
            let mut lpar: Vec<f32> = Vec::new();
            let k = g.len();
            for i in 0..k {
                lpar.push(-xr - 1e-3 * (k - 1 - i) as f32);
            }
            lpar.extend(interior.iter().map(|&i| sk.pos(i).x));
            for i in 0..k {
                lpar.push(xr + 1e-3 * i as f32);
            }
            rows.push(lower.clone());
            lower_rows.push((lower, lpar));
            upper_rows.push((upper, upar));
        }
        for r in 0..rows.len() - 1 {
            let (a, va) = &lower_rows[r];
            let (b, vb) = &upper_rows[r + 1];
            let before = all.len();
            zipper(a, va, b, vb, &mut all);
            shell.cap_tris.extend_from_slice(&all[before..]);
        }
        shell.caps[ci] = rows;
    }
    shell.tris = orient(all, &|i| sk.pos(i));
    let pos = |i: u32| sk.pos(i);
    // Beams: edges, quad braces, bends.
    let mut edges: std::collections::HashMap<(u32, u32), Vec<u32>> = Default::default();
    for t in &shell.tris {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            edges
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(t[(e + 2) % 3]);
        }
    }
    let mut keys_sorted: Vec<(u32, u32)> = edges.keys().copied().collect();
    keys_sorted.sort();
    for &(a, b) in &keys_sorted {
        shell.beams.push((a, b, ShellBeam::Edge));
    }
    for &(a, b) in &keys_sorted {
        let opp = &edges[&(a, b)];
        if opp.len() == 2 {
            let (p, q) = (opp[0], opp[1]);
            let (pp, pq) = (pos(p), pos(q));
            let ab = pos(b) - pos(a);
            // A quad's other diagonal (not across a sliver).
            let d = pp.distance(pq);
            let e = ab.length();
            if d > 0.3 * e && d < 2.5 * e.max(spacing.ring) {
                shell.beams.push((p.min(q), p.max(q), ShellBeam::Brace));
            }
        }
    }
    // Bends along each ring (skip one) and across rings (to the nearest in v two
    // rings on); along cap rows and across them likewise.
    for k in 0..shell.rings.len() {
        let r = &shell.rings[k];
        let mut f = r.half.clone();
        for &i in r.half[1..r.half.len() - 1].iter().rev() {
            f.push(shell.mirror[&i]);
        }
        for i in 0..f.len() {
            shell
                .beams
                .push((f[i], f[(i + 2) % f.len()], ShellBeam::Bend));
        }
        if k + 2 < shell.rings.len() {
            let (a, b) = (&shell.rings[k], &shell.rings[k + 2]);
            for (i, &n) in a.half.iter().enumerate() {
                let j =
                    b.v.iter()
                        .enumerate()
                        .min_by(|x, y| (x.1 - a.v[i]).abs().total_cmp(&(y.1 - a.v[i]).abs()))
                        .unwrap()
                        .0;
                let m = b.half[j];
                shell.beams.push((n, m, ShellBeam::Bend));
                shell
                    .beams
                    .push((shell.mirror[&n], shell.mirror[&m], ShellBeam::Bend));
            }
        }
    }
    for rows in &shell.caps {
        for (r, row) in rows.iter().enumerate() {
            for i in 0..row.len().saturating_sub(2) {
                shell.beams.push((row[i], row[i + 2], ShellBeam::Bend));
            }
            if r + 2 < rows.len() {
                let next = &rows[r + 2];
                for &n in row {
                    let x = pos(n).x;
                    let m = *next
                        .iter()
                        .min_by(|&&p, &&q| (pos(p).x - x).abs().total_cmp(&(pos(q).x - x).abs()))
                        .unwrap();
                    shell.beams.push((n, m, ShellBeam::Bend));
                }
            }
        }
    }
    shell
}

/// Wind every triangle of a closed, connected surface consistently (each edge
/// crossed in opposite directions by its two triangles), outward: the winding is
/// propagated from the triangle with the highest centroid, made to face up.
pub fn orient(tris: Vec<[u32; 3]>, pos: &dyn Fn(u32) -> Vec3) -> Vec<[u32; 3]> {
    let mut tris = tris;
    let mut by_edge: std::collections::HashMap<(u32, u32), Vec<usize>> = Default::default();
    for (ti, t) in tris.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            by_edge.entry((a.min(b), a.max(b))).or_default().push(ti);
        }
    }
    let top = (0..tris.len())
        .max_by(|&a, &b| {
            let ca = tris[a].iter().map(|&i| pos(i).y).sum::<f32>();
            let cb = tris[b].iter().map(|&i| pos(i).y).sum::<f32>();
            ca.total_cmp(&cb)
        })
        .unwrap_or(0);
    let mut done = vec![false; tris.len()];
    let up = |t: &[u32; 3]| {
        let (a, b, c) = (pos(t[0]), pos(t[1]), pos(t[2]));
        (b - a).cross(c - a).y
    };
    if up(&tris[top]) < 0.0 {
        tris[top].swap(1, 2);
    }
    let mut stack = vec![top];
    done[top] = true;
    while let Some(ti) = stack.pop() {
        let t = tris[ti];
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            for &o in &by_edge[&(a.min(b), a.max(b))] {
                if done[o] {
                    continue;
                }
                // The neighbour must cross the edge b → a.
                let u = tris[o];
                let has_ab = (0..3).any(|f| u[f] == a && u[(f + 1) % 3] == b);
                if has_ab {
                    tris[o].swap(1, 2);
                }
                done[o] = true;
                stack.push(o);
            }
        }
    }
    tris
}

/// Closed-surface check: every edge used by exactly two triangles, in opposite
/// directions. Returns the first problem.
pub fn check_closed(tris: &[[u32; 3]]) -> Result<(), String> {
    let mut directed: std::collections::HashMap<(u32, u32), u32> = Default::default();
    for t in tris {
        for e in 0..3 {
            *directed.entry((t[e], t[(e + 1) % 3])).or_default() += 1;
        }
    }
    for (&(a, b), &n) in &directed {
        if n != 1 {
            return Err(format!("edge {a}→{b} used {n} times in one direction"));
        }
        if directed.get(&(b, a)) != Some(&1) {
            return Err(format!(
                "edge {a}→{b} has no opposite (an open or flipped edge)"
            ));
        }
    }
    Ok(())
}

/// Make a sequence strictly increasing (ties nudged), keeping its order.
fn monotone(v: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(v.len());
    for &x in v {
        let x = match out.last() {
            Some(&p) if x <= p => p + 1e-4,
            _ => x,
        };
        out.push(x);
    }
    out
}

/// [`zipper`] for the other builders.
pub fn zipper_pub(a: &[u32], va: &[f32], b: &[u32], vb: &[f32], out: &mut Vec<[u32; 3]>) {
    zipper(a, va, b, vb, out)
}
