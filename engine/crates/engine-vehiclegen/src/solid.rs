//! Solid parts: every panel a closed volume, not a sheet.
//!
//! A panel is its drawn skin (exactly the skin `panels::panel` makes: welded,
//! relaxed, inset for the shut-line gap, corners rounded), an inner face (the
//! skin's own topology pushed below it by a depth field, so every inner
//! vertex has the skin vertex it came from) and edge walls joining the two
//! round every border: the outer perimeter and every opening. The result is
//! closed and manifold by construction. Every vertex carries a
//! [`Surf`] tag: `Outer` (the skin), `Edge` (the walls) or `Inner`.
//!
//! What the edge wall looks like depends on what the border is ([`EdgeKind`])
//! and on the part ([`Style`]):
//! - a shut line against another panel: the rolled hem, then the shut face
//!   (closures and the body), or a return flange (bolted sheet panels);
//! - a glazed opening: a rebate the glass sits in (the run channel of a
//!   door's window frame, the pinch flange of a body aperture);
//! - a wheel arch: the rolled lip and a return towards the liner.
//!
//! The depth field gives the inner face its features: a closure's frame and
//! pockets, a door's belt reinforcement, a bolted panel's thin sheet.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use glam::{DVec2, DVec3};

use crate::fx::{FxHashMap, FxHashSet};
use crate::mesh::{key3, Mat, Mesh, Surf};
use crate::panels::{
    boundary_loops, inset_boundary, loop_dirs, neighbours, py_floordiv, relax, round_corners,
    smooth_normals, Key, HEM_OUT,
};

/// Rows of an edge wall between the skin's border and the inner face's.
pub const WALL_ROWS: usize = 4;
/// The thinnest a solid part may be anywhere (the depth field is clamped to
/// it, and where an offset would fold it is reduced no further than this).
pub const MIN_THICKNESS: f64 = 0.003;
/// Sheet thickness of a bolted panel (a pressing, drawn thicker than its
/// 0.8 mm so the two faces never meet when it bends).
pub const SHEET: f64 = 0.005;
/// Glass slab thickness (toughened side glass and the laminated screens).
pub const GLASS: f64 = 0.0045;
/// Lamp lens thickness.
pub const LENS: f64 = 0.003;

/// What a part is built like.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// A door: skin, shut faces, an inner panel with its frame, a recessed
    /// centre, the belt reinforcement and the window frame's run channel.
    Door,
    /// Bonnet or boot lid: skin and an inner frame (perimeter, spine, braces)
    /// with pockets through which the skin's back shows.
    Lid,
    /// A bolted pressing or moulding (wing, bumper cover): a thin sheet with
    /// return flanges.
    Sheet,
    /// The body shell: a box-section wall, with jambs at every aperture.
    Structure,
}

/// What a border of a panel is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum EdgeKind {
    /// A shut line against another panel.
    Shut = 0,
    /// The edge of a glazed opening (glass or a lamp lens).
    Glazed = 1,
    /// A wheel arch.
    Arch = 2,
    /// Any other free edge (a grille opening, a bumper's lower edge).
    Open = 3,
}

/// An interior vertex of a skin, as the depth field sees it.
pub struct Site {
    pub p: DVec3,
    /// smoothed outward normal
    pub n: DVec3,
    /// distance over the skin to the nearest border (m)
    pub edge: f64,
    /// the inner face's depth at that border
    pub rim: f64,
    /// distance over the skin to the nearest glazed border (infinite if none)
    pub glass: f64,
    /// the skin's bounding box
    pub lo: DVec3,
    pub hi: DVec3,
}

pub struct SolidOpts<'a> {
    pub style: Style,
    /// border vertices (position keys) shared with another panel
    pub shut: Option<&'a FxHashSet<Key>>,
    /// border vertices shared with a glass pane or a lens
    pub glazed: Option<&'a FxHashSet<Key>>,
    /// wheel openings: (centre z, centre y, radius)
    pub arches: &'a [(f64, f64, f64)],
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The edge wall's section for a border kind: [`WALL_ROWS`] points then the
/// inner face's border, each (s, t): s along the skin into the panel (m,
/// negative: out past the edge), t below the skin.
pub fn profile(style: Style, kind: EdgeKind) -> ([(f64, f64); WALL_ROWS], (f64, f64)) {
    use EdgeKind::*;
    use Style::*;
    // the rebate glass sits in: a lip, back under the skin, down past the
    // pane, then a flange out under it
    let rebate = [
        (-0.0012, 0.0035),
        (0.0075, 0.0060),
        (0.0075, 0.0205),
        (-0.0060, 0.0215),
    ];
    match (style, kind) {
        (Door | Lid | Structure, Glazed) => (rebate, (-0.0060, 0.0290)),
        // a lamp opening in a pressing: a plain land, undercut so the lens
        // (which reaches under the skin edge) clears it
        (Sheet, Glazed) => (
            [
                (-0.0012, 0.0035),
                (0.0035, 0.0070),
                (0.0035, 0.0135),
                (0.0035, 0.0200),
            ],
            (0.0035, 0.0260),
        ),
        // the rolled hem, then the shut face
        (Door | Lid, _) => (
            [
                (-0.0025, 0.0040),
                (-0.0012, 0.0085),
                (0.0, 0.0150),
                (0.0, 0.0215),
            ],
            (0.0, 0.0280),
        ),
        (Structure, Arch) => (
            [
                (-0.0025, 0.0040),
                (-0.0040, 0.0160),
                (-0.0070, 0.0300),
                (-0.0100, 0.0440),
            ],
            (-0.0040, 0.0460),
        ),
        (Structure, _) => (
            [
                (-0.0025, 0.0040),
                (-0.0012, 0.0085),
                (0.0, 0.0160),
                (0.0, 0.0230),
            ],
            (0.0, 0.0300),
        ),
        // return flanges: down the outside, across the tip, back up the inside
        (Sheet, Shut) => (
            [
                (-0.0025, 0.0040),
                (-0.0010, 0.0120),
                (0.0, 0.0200),
                (0.0045, 0.0200),
            ],
            (0.0045, SHEET + 0.0005),
        ),
        (Sheet, Arch) => (
            [
                (-0.0025, 0.0040),
                (-0.0040, 0.0200),
                (-0.0080, 0.0400),
                (-0.0035, 0.0400),
            ],
            (0.0045, SHEET + 0.0005),
        ),
        (Sheet, Open) => (
            [
                (-0.0025, 0.0040),
                (-0.0010, 0.0095),
                (0.0, 0.0150),
                (0.0045, 0.0150),
            ],
            (0.0045, SHEET + 0.0005),
        ),
    }
}

/// The inner face's depth below the skin at an interior vertex.
pub fn depth(style: Style, s: &Site) -> f64 {
    let size = s.hi - s.lo;
    let ramp = smoothstep(0.0, 0.035, s.edge);
    match style {
        Style::Door => {
            let (frame, pocket) = (0.050, 0.036);
            // the frame round the perimeter, the belt reinforcement under
            // the window and a beam across the lower panel
            let band = 1.0 - smoothstep(0.075, 0.105, s.edge);
            let belt = 1.0 - smoothstep(0.110, 0.140, s.glass);
            let yb = s.lo.y + 0.21 * size.y;
            let beam = 1.0 - smoothstep(0.022, 0.050, (s.p.y - yb).abs());
            let d = pocket + (frame - pocket) * band.max(belt).max(beam);
            s.rim + (d - s.rim) * ramp
        }
        Style::Lid => {
            let (frame, pocket) = (0.034, 0.009);
            let band = 1.0 - smoothstep(0.060, 0.090, s.edge);
            // a spine down the centre, a cross member and two braces from
            // the centre of the rear edge to the front corners
            let a = (s.p.x - 0.5 * (s.lo.x + s.hi.x)).abs();
            let b = s.p.z - s.lo.z;
            let (hw, len) = (0.5 * size.x, size.z);
            let rib = |dist: f64| 1.0 - smoothstep(0.028, 0.056, dist);
            let spine = rib(a);
            let cross = rib((b - 0.5 * len).abs());
            let (p0, p1) = (DVec2::new(0.0, 0.12 * len), DVec2::new(hw, 0.88 * len));
            let e = (p1 - p0).normalize();
            let r = DVec2::new(a, b) - p0;
            let brace = rib((r - e * r.dot(e)).length());
            let d = pocket + (frame - pocket) * band.max(spine).max(cross).max(brace);
            s.rim + (d - s.rim) * ramp
        }
        // a thick land round a lamp opening (its rebate is cut into it),
        // tapering to the sheet
        Style::Sheet => SHEET + (s.rim - SHEET).max(0.0) * (1.0 - smoothstep(0.012, 0.045, s.edge)),
        Style::Structure => {
            // the floor pan is a single pressing: the cabin sits on it
            let base = if s.n.y < -0.6 { 0.014 } else { 0.030 };
            s.rim + (base - s.rim) * ramp
        }
    }
}

impl SolidOpts<'_> {
    fn kind_at(&self, p: DVec3) -> EdgeKind {
        let k = key3(p, 1e-6);
        if self.glazed.is_some_and(|g| g.contains(&k)) {
            return EdgeKind::Glazed;
        }
        if self.shut.is_some_and(|s| s.contains(&k)) {
            return EdgeKind::Shut;
        }
        for &(zc, yc, r) in self.arches {
            let d = (p.z - zc).hypot(p.y - yc);
            if (d - r).abs() < 0.03 && p.y > yc - 0.6 * r {
                return EdgeKind::Arch;
            }
        }
        EdgeKind::Open
    }
}

/// Distance over the mesh's edges from the source vertices, and each
/// vertex's nearest source.
fn geodesic(m: &Mesh, nb: &[Vec<u32>], src: &[u32]) -> (Vec<f64>, Vec<u32>) {
    let nv = m.v.len();
    let mut dist = vec![f64::INFINITY; nv];
    let mut from = vec![u32::MAX; nv];
    let mut heap: BinaryHeap<(Reverse<u64>, u32)> = BinaryHeap::new();
    for &s in src {
        dist[s as usize] = 0.0;
        from[s as usize] = s;
        heap.push((Reverse(0), s));
    }
    while let Some((Reverse(dk), u)) = heap.pop() {
        let du = dist[u as usize];
        if dk > (du * 1e9) as u64 {
            continue;
        }
        for &w in &nb[u as usize] {
            let d = du + (m.v[w as usize] - m.v[u as usize]).length();
            if d < dist[w as usize] {
                dist[w as usize] = d;
                from[w as usize] = from[u as usize];
                heap.push((Reverse((d * 1e9) as u64), w));
            }
        }
    }
    (dist, from)
}

/// How deep an inward offset may go at each vertex before the surface
/// behind a convex bend folds: 0.65 of the tightest radius of curvature
/// towards a neighbour (infinite where the surface is flat or concave).
fn convex_cap(m: &Mesh, n: &[DVec3], nb: &[Vec<u32>]) -> Vec<f64> {
    (0..m.v.len())
        .map(|i| {
            let mut k = 0.0f64;
            for &j in &nb[i] {
                let e = m.v[j as usize] - m.v[i];
                k = k.max((n[j as usize] - n[i]).dot(e) / e.length_squared().max(1e-18));
            }
            if k > 1e-9 {
                0.65 / k
            } else {
                f64::INFINITY
            }
        })
        .collect()
}

/// A finished solid panel from its blocks: `panels::panel`'s skin, closed
/// with edge walls and an inner face. `block_meshes`: (Mesh, island id).
pub fn solid_panel(block_meshes: &[(Mesh, i32)], gap: f64, island0: i32, o: &SolidOpts) -> Mesh {
    let mut m = Mesh::new();
    for (g, isl) in block_meshes {
        let mut g = g.clone();
        g.set_islands(island0 + isl);
        m.add(&g);
    }
    m.weld(1e-7);
    relax(&mut m, 6, 0.35, 12.0);
    let loops = boundary_loops(&m);
    // what each border vertex is, from the drawn border (before the inset)
    let mut kind = vec![EdgeKind::Open; m.v.len()];
    for lp in &loops {
        for &i in lp {
            kind[i as usize] = o.kind_at(m.v[i as usize]);
        }
    }
    let d0 = HEM_OUT + gap / 2.0;
    let d: Vec<f64> = match o.shut {
        Some(shut) => {
            m.v.iter()
                .map(|p| {
                    if shut.contains(&key3(*p, 1e-6)) {
                        d0
                    } else {
                        HEM_OUT * 0.6
                    }
                })
                .collect()
        }
        None => vec![d0; m.v.len()],
    };
    let (loops, isb) = inset_boundary(&mut m, &d, None, Some(loops));
    round_corners(&mut m, &loops, 35.0);
    shell(&m, &loops, &isb, &kind, island0, o.style)
}

struct Rim {
    /// in-surface inward direction (miter scaled)
    w: DVec3,
    rows: [(f64, f64); WALL_ROWS],
    inner: (f64, f64),
    /// the wall's depth scale (reduced where the offset would fold)
    f: f64,
}

/// Skin + edge walls + inner face.
fn shell(
    skin: &Mesh,
    loops: &[Vec<u32>],
    isb: &[bool],
    kind: &[EdgeKind],
    island0: i32,
    style: Style,
) -> Mesh {
    let nv = skin.v.len();
    let n = smooth_normals(skin, 6);
    let nb = neighbours(skin);
    let loops: Vec<&Vec<u32>> = loops.iter().filter(|l| l.len() >= 3).collect();
    // the wall section at every border vertex: its kind's, blended along
    // the loop where the kind changes
    let mut rim: FxHashMap<u32, Rim> = FxHashMap::default();
    for lp in &loops {
        let l = lp.len();
        let w = loop_dirs(skin, lp, &n, 0);
        let mut wt: Vec<[f64; 4]> = lp
            .iter()
            .map(|&i| {
                let mut a = [0.0; 4];
                a[kind[i as usize] as usize] = 1.0;
                a
            })
            .collect();
        for _ in 0..2 {
            wt = (0..l)
                .map(|i| {
                    std::array::from_fn(|k| {
                        0.25 * wt[(i + l - 1) % l][k] + 0.5 * wt[i][k] + 0.25 * wt[(i + 1) % l][k]
                    })
                })
                .collect();
        }
        let seg: Vec<f64> = (0..l)
            .map(|i| (skin.v[lp[(i + 1) % l] as usize] - skin.v[lp[i] as usize]).length())
            .collect();
        // how sharply the border turns at each vertex (0: under 15 degrees,
        // 1: 60 and over), spread to its neighbours
        let turn: Vec<f64> = (0..l)
            .map(|i| {
                let p = |k: usize| skin.v[lp[k % l] as usize];
                let (t0, t1) = (
                    (p(i) - p(i + l - 1)).normalize_or_zero(),
                    (p(i + 1) - p(i)).normalize_or_zero(),
                );
                ((t0.dot(t1).clamp(-1.0, 1.0).acos().to_degrees() - 15.0) / 45.0).clamp(0.0, 1.0)
            })
            .collect();
        let sharp = |i: usize| {
            let t = |k: isize| turn[(i as isize + k).rem_euclid(l as isize) as usize];
            t(0).max(0.7 * t(1).max(t(-1))).max(0.4 * t(2).max(t(-2)))
        };
        for i in 0..l {
            // short border segments and tight corners take a narrower
            // section (its rows would cross inside the corner)
            let sc = (seg[i].min(seg[(i + l - 1) % l]) / 0.012).min(1.0) * (1.0 - 0.85 * sharp(i));
            let mut rows = [(0.0, 0.0); WALL_ROWS];
            let mut inner = (0.0, 0.0);
            for (k, kd) in [
                EdgeKind::Shut,
                EdgeKind::Glazed,
                EdgeKind::Arch,
                EdgeKind::Open,
            ]
            .into_iter()
            .enumerate()
            {
                let (r, q) = profile(style, kd);
                for j in 0..WALL_ROWS {
                    rows[j].0 += wt[i][k] * r[j].0 * sc;
                    rows[j].1 += wt[i][k] * r[j].1;
                }
                inner.0 += wt[i][k] * q.0 * sc;
                inner.1 += wt[i][k] * q.1;
            }
            rim.insert(
                lp[i],
                Rim {
                    w: w[i],
                    rows,
                    inner,
                    f: 1.0,
                },
            );
        }
    }
    // the depth field
    let border: Vec<u32> = (0..nv as u32).filter(|&i| isb[i as usize]).collect();
    let glazed: Vec<u32> = border
        .iter()
        .copied()
        .filter(|&i| kind[i as usize] == EdgeKind::Glazed)
        .collect();
    let (edge, from) = geodesic(skin, &nb, &border);
    let glass = if glazed.is_empty() {
        vec![f64::INFINITY; nv]
    } else {
        geodesic(skin, &nb, &glazed).0
    };
    let (lo, hi) = skin.bbox();
    let mut d: Vec<f64> = (0..nv)
        .map(|i| {
            if isb[i] {
                return 0.0;
            }
            let r = rim.get(&from[i]).map_or(0.02, |r| r.inner.1);
            depth(
                style,
                &Site {
                    p: skin.v[i],
                    n: n[i],
                    edge: edge[i],
                    rim: r,
                    glass: glass[i],
                    lo,
                    hi,
                },
            )
            .max(MIN_THICKNESS)
        })
        .collect();
    // behind a convex bend an offset surface shrinks, and folds where it is
    // deeper than the bend's radius: no deeper than 0.65 of it there, the
    // limit eased into the neighbourhood
    let designed = d.clone();
    let cap = convex_cap(skin, &n, &nb);
    for i in 0..nv {
        match rim.get_mut(&(i as u32)) {
            Some(r) => r.f = (cap[i] / r.inner.1).min(1.0),
            None => d[i] = d[i].min(cap[i]).max(MIN_THICKNESS),
        }
    }
    for _ in 0..3 {
        let prev = d.clone();
        for i in 0..nv {
            if !isb[i] && !nb[i].is_empty() {
                let mean = nb[i]
                    .iter()
                    .map(|&j| {
                        if isb[j as usize] {
                            rim.get(&j).map_or(prev[i], |r| r.inner.1 * r.f)
                        } else {
                            prev[j as usize]
                        }
                    })
                    .sum::<f64>()
                    / nb[i].len() as f64;
                d[i] = d[i].min(1.25 * mean).max(MIN_THICKNESS);
            }
        }
    }
    // inner positions; where the offset still squeezes or folds a face the
    // depth is reduced there until it does not
    let place = |d: &[f64], rim: &FxHashMap<u32, Rim>| -> Vec<DVec3> {
        (0..nv)
            .map(|i| match rim.get(&(i as u32)) {
                Some(r) => {
                    skin.v[i] + r.w * r.inner.0 - n[i] * (r.inner.1 * r.f).max(MIN_THICKNESS)
                }
                None => skin.v[i] - n[i] * d[i],
            })
            .collect()
    };
    let faces = skin.faces();
    let mut q = place(&d, &rim);
    for _ in 0..60 {
        let mut bad = vec![false; nv];
        let mut any = false;
        for f in &faces {
            let k = f.len();
            let p = |j: usize| skin.v[f[j % k] as usize];
            let g = |j: usize| q[f[j % k] as usize];
            let mut squeezed = false;
            for j in 0..k {
                // every corner keeps its turn, every edge a third of its length
                let no = (p(j + 1) - p(j)).cross(p(j + k - 1) - p(j));
                let ni = (g(j + 1) - g(j)).cross(g(j + k - 1) - g(j));
                if ni.dot(no) < 0.2 * ni.length() * no.length()
                    || (g(j + 1) - g(j)).length() < 0.34 * (p(j + 1) - p(j)).length()
                {
                    squeezed = true;
                    break;
                }
            }
            if squeezed {
                any = true;
                for &i in f {
                    bad[i as usize] = true;
                }
            }
        }
        if !any {
            break;
        }
        for i in 0..nv {
            if !bad[i] {
                continue;
            }
            match rim.get_mut(&(i as u32)) {
                Some(r) => r.f = (r.f * 0.88).max(MIN_THICKNESS / r.inner.1.max(MIN_THICKNESS)),
                None => d[i] = (d[i] * 0.88).max(MIN_THICKNESS),
            }
        }
        q = place(&d, &rim);
    }
    // a wall no deeper than the inner face beside it (its rows would come
    // through it where the depth was reduced)
    let ids: Vec<u32> = rim.keys().copied().collect();
    for i in ids {
        let mut lim = f64::INFINITY;
        for &j in &nb[i as usize] {
            if !isb[j as usize] {
                if d[j as usize] < designed[j as usize] - 1e-4 {
                    lim = lim.min(d[j as usize]);
                }
                for &k in &nb[j as usize] {
                    if !isb[k as usize] && d[k as usize] < designed[k as usize] - 1e-4 {
                        lim = lim.min(d[k as usize]);
                    }
                }
            }
        }
        let r = rim.get_mut(&i).unwrap();
        let tmax = r.rows.iter().map(|x| x.1).fold(r.inner.1, f64::max);
        if lim.is_finite() {
            r.f = r.f.min((lim - 0.0015) / tmax).max(MIN_THICKNESS / tmax);
        }
    }
    let q = place(&d, &rim);
    // assemble: skin, walls, inner face
    let mat = skin
        .qm
        .first()
        .or(skin.tm.first())
        .copied()
        .unwrap_or(Mat::Paint);
    let mut out = skin.clone();
    let mut surf = vec![Surf::Outer; nv];
    let nrows: usize = loops.iter().map(|l| l.len() * WALL_ROWS).sum();
    let inner_of = |i: u32| (nv + nrows) as u32 + i;
    let mut island = island0 + 100;
    let chunk = 1.2;
    for lp in &loops {
        let l = lp.len();
        let base = out.v.len() as u32;
        let row = |k: usize, i: usize| base + (k * l + i) as u32;
        let mut step = [0.0f64; WALL_ROWS + 1];
        for k in 0..WALL_ROWS {
            for (i, &vi) in lp.iter().enumerate() {
                let r = &rim[&vi];
                let (s, t) = r.rows[k];
                let t = t * r.f;
                let x = skin.v[vi as usize] + r.w * s - n[vi as usize] * t;
                let prev = if k == 0 {
                    skin.v[vi as usize]
                } else {
                    out.v[row(k - 1, i) as usize]
                };
                step[k] += (x - prev).length() / l as f64;
                if k == WALL_ROWS - 1 {
                    step[k + 1] += (q[vi as usize] - x).length() / l as f64;
                }
                out.v.push(x);
            }
        }
        let mut vv = [0.0f64; WALL_ROWS + 2];
        for k in 0..=WALL_ROWS {
            vv[k + 1] = vv[k] + step[k];
        }
        let mut s = vec![0.0];
        for i in 0..l {
            let x = s[i] + (skin.v[lp[(i + 1) % l] as usize] - skin.v[lp[i] as usize]).length();
            s.push(x);
        }
        for i in 0..l {
            let j = (i + 1) % l;
            let cf = py_floordiv(s[i], chunk);
            let (ua, ub) = (s[i] - cf * chunk, s[i + 1] - cf * chunk);
            for k in 0..=WALL_ROWS {
                // the ring above (towards the skin) and the ring below
                let (a0, b0) = if k == 0 {
                    (lp[i], lp[j])
                } else {
                    (row(k - 1, i), row(k - 1, j))
                };
                let (a1, b1) = if k == WALL_ROWS {
                    (inner_of(lp[i]), inner_of(lp[j]))
                } else {
                    (row(k, i), row(k, j))
                };
                out.q.push([b0, a0, a1, b1]);
                out.quv.push([
                    DVec2::new(ub, vv[k]),
                    DVec2::new(ua, vv[k]),
                    DVec2::new(ua, vv[k + 1]),
                    DVec2::new(ub, vv[k + 1]),
                ]);
                out.qi.push(island + cf as i32);
                out.qm.push(mat);
            }
        }
        island += py_floordiv(s[l], chunk) as i32 + 1;
    }
    surf.resize(out.v.len(), Surf::Edge);
    debug_assert_eq!(out.v.len(), nv + nrows);
    out.v.extend_from_slice(&q);
    surf.resize(out.v.len(), Surf::Inner);
    let isl = |x: i32| if x >= 0 { island0 + 1000 + x } else { -1 };
    for (k, f) in skin.q.iter().enumerate() {
        let i = f.map(inner_of);
        let g = [i[0], i[3], i[2], i[1]];
        let u = skin.quv[k];
        let uv = [u[0], u[3], u[2], u[1]];
        out.q.push(g);
        out.quv.push(uv);
        out.qi.push(isl(skin.qi[k]));
        out.qm.push(skin.qm[k]);
    }
    for (k, f) in skin.t.iter().enumerate() {
        let mut g = f.map(inner_of);
        g.reverse();
        let mut uv = skin.tuv[k];
        uv.reverse();
        out.t.push(g);
        out.tuv.push(uv);
        out.ti.push(isl(skin.ti[k]));
        out.tm.push(skin.tm[k]);
    }
    if let Some(c) = &mut out.col {
        c.resize(nv + nrows, [1.0; 4]);
        let src = skin.col.as_ref().unwrap();
        c.extend_from_slice(src);
    }
    out.surf = Some(surf);
    out
}

/// A single-sided surface as a closed slab `th` thick: the surface, its copy
/// pushed back along the vertex normals and a wall round every border.
pub fn slab(m: &Mesh, th: f64) -> Mesh {
    slab_with(m, th, false)
}

/// [`slab`] for a moulding that wraps a corner (a lamp lens): its back is the
/// front moved along one draw direction (the mean normal), as a moulded
/// lens is, so the back cannot cross the front at the crease.
pub fn slab_drawn(m: &Mesh, th: f64) -> Mesh {
    slab_with(m, th, true)
}

fn slab_with(m: &Mesh, th: f64, drawn: bool) -> Mesh {
    let nv = m.v.len() as u32;
    let mut out = m.clone();
    if drawn {
        let n = m.vertex_normals();
        let dir = n.iter().copied().sum::<DVec3>().normalize_or_zero();
        // one move for every vertex (a varying one would cross where the
        // surface runs along the draw): 1.4 th, so the lens is th thick
        // where it faces 45 degrees off the draw
        out.v.extend(m.v.iter().map(|p| *p - dir * (1.4 * th)));
    } else {
        let n = smooth_normals(m, 8);
        let cap = convex_cap(m, &n, &neighbours(m));
        // thinner (down to a third) behind a bend tighter than the slab is thick
        out.v
            .extend((0..m.v.len()).map(|i| m.v[i] - n[i] * th.min(cap[i]).max(th / 3.0)));
    }
    let mut surf = vec![Surf::Outer; nv as usize];
    surf.resize(out.v.len(), Surf::Inner);
    for (k, f) in m.q.iter().enumerate() {
        let i = f.map(|i| i + nv);
        let g = [i[0], i[3], i[2], i[1]];
        let u = m.quv[k];
        let uv = [u[0], u[3], u[2], u[1]];
        out.q.push(g);
        out.quv.push(uv);
        out.qi.push(m.qi[k]);
        out.qm.push(m.qm[k]);
    }
    for (k, f) in m.t.iter().enumerate() {
        let mut g = f.map(|i| i + nv);
        g.reverse();
        let mut uv = m.tuv[k];
        uv.reverse();
        out.t.push(g);
        out.tuv.push(uv);
        out.ti.push(m.ti[k]);
        out.tm.push(m.tm[k]);
    }
    let mat = m.qm.first().or(m.tm.first()).copied().unwrap_or(Mat::Paint);
    for lp in boundary_loops(m) {
        let l = lp.len();
        if l < 3 {
            continue;
        }
        for i in 0..l {
            let (a, b) = (lp[i], lp[(i + 1) % l]);
            out.q.push([b, a, a + nv, b + nv]);
            out.quv.push([DVec2::ZERO; 4]);
            out.qi.push(-1);
            out.qm.push(mat);
        }
    }
    if let Some(c) = &mut out.col {
        let src = c.clone();
        c.extend(src);
    }
    out.surf = Some(surf);
    out
}

// -- checks ----------------------------------------------------------------------------

/// What [`check`] measures on a part.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// edges with no opposite (an open border)
    pub boundary_edges: usize,
    /// edges used more than once in one direction, or by more than two faces
    pub nonmanifold_edges: usize,
    /// signed volume (m^3): positive when the faces look outward
    pub volume: f64,
    /// closed components
    pub components: usize,
    /// the least signed volume of a component
    pub min_component_volume: f64,
    /// inner-face vertices' distance to the nearest skin vertex (m)
    pub thickness_min: f64,
    pub thickness_max: f64,
    pub triangles: usize,
}

impl Report {
    pub fn closed(&self) -> bool {
        self.boundary_edges == 0 && self.nonmanifold_edges == 0
    }
}

pub fn check(m: &Mesh) -> Report {
    let tris = crate::export::split_triangles(m);
    let mut dir: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    for f in m.faces() {
        let k = f.len();
        for i in 0..k {
            *dir.entry((f[i], f[(i + 1) % k])).or_insert(0) += 1;
        }
    }
    let mut r = Report {
        triangles: tris.len(),
        ..Default::default()
    };
    for (&(a, b), &c) in &dir {
        if c > 1 {
            r.nonmanifold_edges += 1;
        }
        if !dir.contains_key(&(b, a)) {
            r.boundary_edges += 1;
        }
    }
    // components (union-find over faces' vertices) and their volumes
    let mut parent: Vec<u32> = (0..m.v.len() as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for f in &tris {
        let a = find(&mut parent, f[0]);
        for &j in &f[1..] {
            let b = find(&mut parent, j);
            parent[b as usize] = a;
        }
    }
    let mut vol: FxHashMap<u32, f64> = FxHashMap::default();
    for f in &tris {
        let (a, b, c) = (m.v[f[0] as usize], m.v[f[1] as usize], m.v[f[2] as usize]);
        let v = a.dot(b.cross(c)) / 6.0;
        r.volume += v;
        *vol.entry(find(&mut parent, f[0])).or_insert(0.0) += v;
    }
    r.components = vol.len();
    r.min_component_volume = vol.values().copied().fold(f64::INFINITY, f64::min);
    // thickness: every inner vertex to the nearest skin vertex
    let cell = 0.03;
    let key = |p: DVec3| {
        (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        )
    };
    let mut grid: FxHashMap<(i32, i32, i32), Vec<u32>> = FxHashMap::default();
    let mut used = vec![false; m.v.len()];
    for f in &tris {
        for &i in f {
            used[i as usize] = true;
        }
    }
    for (i, p) in m.v.iter().enumerate() {
        if used[i] && m.surf_of(i) == Surf::Outer {
            grid.entry(key(*p)).or_default().push(i as u32);
        }
    }
    r.thickness_min = f64::INFINITY;
    for (i, p) in m.v.iter().enumerate() {
        if !used[i] || m.surf_of(i) != Surf::Inner {
            continue;
        }
        let k = key(*p);
        let mut best = f64::INFINITY;
        for dx in -3..=3 {
            for dy in -3..=3 {
                for dz in -3..=3 {
                    if let Some(c) = grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        for &j in c {
                            best = best.min((m.v[j as usize] - *p).length());
                        }
                    }
                }
            }
        }
        r.thickness_min = r.thickness_min.min(best);
        r.thickness_max = r.thickness_max.max(best);
    }
    r
}

/// Faces of a part that cut through each other: edges of every `stride`-th
/// triangle against the triangles near them (sharing no vertex). Returns the
/// crossings' points.
pub fn self_intersections(m: &Mesh, stride: usize) -> Vec<DVec3> {
    crossings(m, stride).into_iter().map(|c| c.3).collect()
}

/// [`self_intersections`] with the edge and the triangle of each crossing.
pub fn crossings(m: &Mesh, stride: usize) -> Vec<(u32, u32, [u32; 3], DVec3)> {
    let tris = crate::export::split_triangles(m);
    let comp = components(m);
    let cell = 0.04;
    let ck = |x: f64| (x / cell).floor() as i32;
    let mut grid: FxHashMap<(i32, i32, i32), Vec<u32>> = FxHashMap::default();
    for (ti, f) in tris.iter().enumerate() {
        let p = f.map(|i| m.v[i as usize]);
        let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
        for x in ck(lo.x)..=ck(hi.x) {
            for y in ck(lo.y)..=ck(hi.y) {
                for z in ck(lo.z)..=ck(hi.z) {
                    grid.entry((x, y, z)).or_default().push(ti as u32);
                }
            }
        }
    }
    let mut hits = Vec::new();
    let mut seen: FxHashSet<(u32, u32, u32)> = FxHashSet::default();
    for (ti, f) in tris.iter().enumerate().step_by(stride.max(1)) {
        let _ = ti;
        for e in 0..3 {
            let (ia, ib) = (f[e], f[(e + 1) % 3]);
            let (a, b) = (m.v[ia as usize], m.v[ib as usize]);
            let (lo, hi) = (a.min(b), a.max(b));
            for x in ck(lo.x)..=ck(hi.x) {
                for y in ck(lo.y)..=ck(hi.y) {
                    for z in ck(lo.z)..=ck(hi.z) {
                        let Some(c) = grid.get(&(x, y, z)) else {
                            continue;
                        };
                        for &tj in c {
                            let g = tris[tj as usize];
                            if comp[g[0] as usize] != comp[ia as usize]
                                || g.contains(&ia)
                                || g.contains(&ib)
                                || !seen.insert((ia.min(ib), ia.max(ib), tj))
                            {
                                continue;
                            }
                            let (p0, p1, p2) =
                                (m.v[g[0] as usize], m.v[g[1] as usize], m.v[g[2] as usize]);
                            // coincident points of separate pieces (a part
                            // of several solids touching) are not crossings
                            if [p0, p1, p2]
                                .iter()
                                .any(|p| (*p - a).length() < 1e-7 || (*p - b).length() < 1e-7)
                            {
                                continue;
                            }
                            if let Some(x) = segment_triangle(a, b, p0, p1, p2) {
                                hits.push((ia, ib, g, x));
                            }
                        }
                    }
                }
            }
        }
    }
    hits
}

/// Each vertex's connected piece (a label per vertex).
fn components(m: &Mesh) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..m.v.len() as u32).collect();
    fn find(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for f in m.faces() {
        let a = find(&mut parent, f[0]);
        for &j in &f[1..] {
            let b = find(&mut parent, j);
            parent[b as usize] = a;
        }
    }
    (0..m.v.len() as u32)
        .map(|i| find(&mut parent, i))
        .collect()
}

/// Where the segment a-b passes through the triangle's interior.
fn segment_triangle(a: DVec3, b: DVec3, p0: DVec3, p1: DVec3, p2: DVec3) -> Option<DVec3> {
    let (e1, e2, d) = (p1 - p0, p2 - p0, b - a);
    let h = d.cross(e2);
    let det = e1.dot(h);
    let scale = e1.length() * e2.length() * d.length();
    if det.abs() < 1e-9 * scale.max(1e-30) {
        return None;
    }
    let s = a - p0;
    let u = s.dot(h) / det;
    let qv = s.cross(e1);
    let v = d.dot(qv) / det;
    let t = e2.dot(qv) / det;
    let eps = 1e-6;
    (u > eps && v > eps && u + v < 1.0 - eps && t > eps && t < 1.0 - eps).then(|| a + d * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{grid, Grid2};

    /// A curved patch with a hole (a door with its window opening).
    fn patch() -> Vec<(Mesh, i32)> {
        let f = |i: usize, j: usize| {
            let (x, z) = (0.02 * i as f64, 0.02 * j as f64);
            DVec3::new(x, 0.3 + 0.1 * (1.3 * x).sin() * (0.9 * z).cos(), z)
        };
        let mut out = Vec::new();
        // a 40 x 40 cell patch as a ring of four blocks round a 16 x 16 hole
        for (k, (i0, i1, j0, j1)) in [
            (0, 13, 0, 41),
            (28, 41, 0, 41),
            (12, 29, 0, 13),
            (12, 29, 28, 41),
        ]
        .into_iter()
        .enumerate()
        {
            let g = Grid2::from_fn(i1 - i0, j1 - j0, |a, b| f(i0 + a, j0 + b));
            let uv = Grid2::from_fn(i1 - i0, j1 - j0, |a, b| {
                DVec2::new(0.02 * (i0 + a) as f64, 0.02 * (j0 + b) as f64)
            });
            // (dP/du x dP/dv) points down for x then z: wind the other way
            out.push((grid(&g, Some(&uv), Mat::Paint, false, false, -1), k as i32));
        }
        out
    }

    #[test]
    fn every_style_closes_a_patch_with_a_hole() {
        for style in [Style::Door, Style::Lid, Style::Sheet, Style::Structure] {
            let m = solid_panel(
                &patch(),
                0.0046,
                0,
                &SolidOpts {
                    style,
                    shut: None,
                    glazed: None,
                    arches: &[],
                },
            );
            let r = check(&m);
            assert!(r.closed(), "{style:?}: {r:?}");
            assert_eq!(r.components, 1, "{style:?}");
            assert!(r.volume.abs() > 1e-4, "{style:?}: {r:?}");
            assert!(
                r.thickness_min >= MIN_THICKNESS - 1e-9 && r.thickness_max < 0.07,
                "{style:?}: {r:?}"
            );
            assert!(self_intersections(&m, 1).is_empty(), "{style:?}");
            let surf = m.surf.as_ref().unwrap();
            assert_eq!(surf.len(), m.v.len());
            // one inner vertex per skin vertex
            let count = |s: Surf| surf.iter().filter(|x| **x == s).count();
            assert_eq!(count(Surf::Outer), count(Surf::Inner));
            assert!(count(Surf::Edge) > 0);
            // the skin alone is the sheet it was
            let sk = m.outer_only();
            assert_eq!(sk.v.len(), count(Surf::Outer));
            assert_eq!(boundary_loops(&sk).len(), 2);
        }
    }

    #[test]
    fn a_slab_is_closed_and_as_thick_as_asked() {
        let g = Grid2::from_fn(8, 6, |i, j| {
            DVec3::new(0.02 * i as f64, 0.0, -0.02 * j as f64)
        });
        let m = slab(&grid(&g, None, Mat::Glass, false, false, -1), GLASS);
        let r = check(&m);
        assert!(r.closed() && r.components == 1, "{r:?}");
        let want = 0.14 * 0.10 * GLASS;
        assert!((r.volume.abs() - want).abs() < 1e-9, "{r:?}");
        assert!((r.thickness_min - GLASS).abs() < 1e-12 && (r.thickness_max - GLASS).abs() < 1e-12);
    }

    #[test]
    fn crossing_faces_are_found() {
        let g = Grid2::from_fn(3, 3, |i, j| DVec3::new(0.1 * i as f64, 0.0, 0.1 * j as f64));
        let mut m = grid(&g, None, Mat::Paint, false, false, -1);
        assert!(self_intersections(&m, 1).is_empty());
        let h = Grid2::from_fn(2, 2, |i, j| {
            DVec3::new(0.05 + 0.02 * i as f64, -0.05 + 0.1 * j as f64, 0.07)
        });
        m.add(&grid(&h, None, Mat::Paint, false, false, -1));
        // separate pieces of a part may overlap (trim laid on the body) ...
        assert!(self_intersections(&m, 1).is_empty());
        // ... one piece may not cut through itself
        m.push_tris(&[[0, 9, 10]], Mat::Paint);
        assert!(!self_intersections(&m, 1).is_empty());
    }
}
