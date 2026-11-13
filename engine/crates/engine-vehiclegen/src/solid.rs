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
        (-0.0015, 0.0055),
        (0.0075, 0.0075),
        (0.0075, 0.0205),
        (-0.0060, 0.0215),
    ];
    // (every step of a section is at least 5.2 mm: a wall row is no thinner
    // than a 27 mm border segment over ROW_ASPECT)
    match (style, kind) {
        (Door | Lid | Structure, Glazed) => (rebate, (-0.0060, 0.0290)),
        // a lamp opening in a pressing: a plain land, undercut so the lens
        // (which reaches under the skin edge) clears it
        (Sheet, Glazed) => (
            [
                (-0.0015, 0.0055),
                (0.0065, 0.0085),
                (0.0065, 0.0145),
                (0.0065, 0.0205),
            ],
            (0.0065, 0.0260),
        ),
        // a lid's free edge (the boot lid's, over the tail panel): the hem
        // and a flat return, shallow enough to close over the body's edge
        (Lid, Open) => (
            [
                (-0.0026, 0.0045),
                (0.0015, 0.0090),
                (0.0075, 0.0090),
                (0.0135, 0.0090),
            ],
            (0.0195, 0.0090),
        ),
        // the rolled hem, then the shut face
        (Door | Lid, _) => (
            [
                (-0.0025, 0.0048),
                (-0.0012, 0.0102),
                (0.0, 0.0161),
                (0.0, 0.0220),
            ],
            (0.0, 0.0280),
        ),
        (Structure, Arch) => (
            [
                (-0.0025, 0.0048),
                (-0.0040, 0.0160),
                (-0.0070, 0.0300),
                (-0.0100, 0.0440),
            ],
            (-0.0040, 0.0460),
        ),
        (Structure, _) => (
            [
                (-0.0025, 0.0048),
                (-0.0012, 0.0102),
                (0.0, 0.0168),
                (0.0, 0.0234),
            ],
            (0.0, 0.0300),
        ),
        // return flanges: down the outside, across the tip, back up the inside
        (Sheet, Shut) => (
            [
                (-0.0025, 0.0048),
                (-0.0010, 0.0120),
                (0.0, 0.0200),
                (0.0055, 0.0200),
            ],
            (0.0055, SHEET + 0.0005),
        ),
        (Sheet, Arch) => (
            [
                (-0.0025, 0.0048),
                (-0.0040, 0.0200),
                (-0.0080, 0.0400),
                (-0.0025, 0.0400),
            ],
            (0.0055, SHEET + 0.0005),
        ),
        (Sheet, Open) => (
            [
                (-0.0025, 0.0048),
                (-0.0010, 0.0102),
                (0.0, 0.0160),
                (0.0055, 0.0160),
            ],
            (0.0055, SHEET + 0.0005),
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
            let band = 1.0 - smoothstep(0.070, 0.115, s.edge);
            let belt = 1.0 - smoothstep(0.105, 0.150, s.glass);
            let yb = s.lo.y + 0.21 * size.y;
            let beam = 1.0 - smoothstep(0.018, 0.060, (s.p.y - yb).abs());
            let d = pocket + (frame - pocket) * band.max(belt).max(beam);
            s.rim + (d - s.rim) * ramp
        }
        Style::Lid => {
            let (frame, pocket) = (0.034, 0.009);
            let band = 1.0 - smoothstep(0.052, 0.098, s.edge);
            let a = (s.p.x - 0.5 * (s.lo.x + s.hi.x)).abs();
            let b = s.p.z - s.lo.z;
            let (hw, len) = (0.5 * size.x, size.z);
            let rib = |dist: f64| 1.0 - smoothstep(0.020, 0.066, dist);
            let spine = rib(a);
            let ribs = if len > 0.9 {
                // a bonnet: a cross member and two braces from the centre
                // of the rear edge to the front corners
                let cross = rib((b - 0.5 * len).abs());
                let (p0, p1) = (DVec2::new(0.0, 0.12 * len), DVec2::new(hw, 0.88 * len));
                let e = (p1 - p0).normalize();
                let r = DVec2::new(a, b) - p0;
                cross.max(rib((r - e * r.dot(e)).length()))
            } else {
                // a boot lid: a rib each side, half way out
                rib((a - 0.5 * hw).abs())
            };
            let d = pocket + (frame - pocket) * band.max(spine).max(ribs);
            // behind a shallow free edge the frame starts further in
            let ramp = if s.rim < 0.012 {
                smoothstep(0.030, 0.085, s.edge)
            } else {
                ramp
            };
            s.rim + (d - s.rim) * ramp
        }
        Style::Sheet => {
            // a thick land round a lamp opening (its rebate is cut into it),
            // tapering to the sheet
            let land = (s.rim - SHEET).max(0.0) * (1.0 - smoothstep(0.012, 0.045, s.edge));
            // a mount flange: the sheet doubled in a band along every edge
            let flange =
                0.004 * (1.0 - smoothstep(0.024, 0.048, s.edge)) * smoothstep(0.0, 0.018, s.edge);
            // stiffening ribs, clear of the edges
            let rib = |dist: f64| 1.0 - smoothstep(0.008, 0.034, dist);
            let t = (s.p.y - s.lo.y) / size.y.max(1e-9);
            let ribs = if size.x > 1.0 {
                // a bumper cover: the pocket the bumper beam sits in (two
                // rails across the middle) and two webs down each cheek
                let a = (s.p.x - 0.5 * (s.lo.x + s.hi.x)).abs();
                let hw = 0.5 * size.x;
                let (y0, y1) = (s.lo.y + 0.58 * size.y, s.lo.y + 0.86 * size.y);
                let across = 1.0 - smoothstep(0.62 * hw, 0.62 * hw + 0.03, a);
                let rails = rib((s.p.y - y0).abs()).max(rib((s.p.y - y1).abs())) * across;
                let webs = rib((a - 0.62 * hw).abs()).max(rib((a - 0.50 * hw).abs()));
                0.022 * rails.max(webs)
            } else {
                // a wing: a brace under the top flange and two stiffeners
                // down it
                let brace = rib((t - 0.80).abs() * size.y);
                let (z0, z1) = (s.lo.z + 0.22 * size.z, s.lo.z + 0.80 * size.z);
                let down = rib((s.p.z - z0).abs()).max(rib((s.p.z - z1).abs()))
                    * smoothstep(0.30, 0.40, t);
                0.016 * brace.max(down)
            };
            SHEET
                + land
                    .max(flange)
                    .max(ribs * smoothstep(0.024, 0.050, s.edge))
        }
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
    /// the longer of the two border segments at the vertex
    seg: f64,
}

/// The skin's neighbours plus the far corner of every quad (distances over
/// the mesh then run diagonally too, not only along the grid).
fn neighbours8(m: &Mesh, nb: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut o = nb.to_vec();
    for f in &m.q {
        o[f[0] as usize].push(f[2]);
        o[f[2] as usize].push(f[0]);
        o[f[1] as usize].push(f[3]);
        o[f[3] as usize].push(f[1]);
    }
    for s in &mut o {
        s.sort_unstable();
        s.dedup();
    }
    o
}

/// Lower `d` until it rises no faster than `slope` per metre over the mesh
/// from any of the `seeds` (all vertices: the whole field).
fn limit_slope(m: &Mesh, nb: &[Vec<u32>], d: &mut [f64], slope: f64, seeds: &[u32]) {
    let key = |x: f64| Reverse((x * 1e9) as u64);
    let mut heap: BinaryHeap<(Reverse<u64>, u32)> =
        seeds.iter().map(|&i| (key(d[i as usize]), i)).collect();
    while let Some((Reverse(dk), u)) = heap.pop() {
        let du = d[u as usize];
        if dk > (du * 1e9) as u64 {
            continue;
        }
        for &w in &nb[u as usize] {
            let x = du + slope * (m.v[w as usize] - m.v[u as usize]).length();
            if x < d[w as usize] - 1e-9 {
                d[w as usize] = x;
                heap.push((key(x), w));
            }
        }
    }
}

/// The steepest the inner face's depth may change over the skin (a 42
/// degree slope: features stay features, a depth reduced at one place to
/// keep the face from folding spreads into a dish, not a spike).
const SLOPE: f64 = 0.9;
/// A wall row is at least a border segment over this (the art spec's 6:1
/// triangle aspect, with room for skew).
const ROW_ASPECT: f64 = 5.3;
/// Passes evening out the inner face's cells along the surface.
const EVEN_PASSES: usize = 4;

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
    let mut n = smooth_normals(skin, 6);
    let nb = neighbours(skin);
    let nb8 = neighbours8(skin, &nb);
    let faces = skin.faces();
    // A thin cell of the skin (a row a quarter as wide as it is long: a
    // tight bend drawn as its own row, the rows of a narrow pillar) has no
    // width to lose: offset along normals that converge it would be a
    // sliver. Thin rows are offset along one direction across their width
    // (the normals averaged along the thin edges), and the wider cells
    // beside them take up the bend.
    let mut across: Vec<(u32, u32)> = Vec::new();
    for f in &faces {
        let k = f.len();
        let len = |j: usize| (skin.v[f[(j + 1) % k] as usize] - skin.v[f[j % k] as usize]).length();
        let long = (0..k).map(len).fold(0.0, f64::max);
        for j in 0..k {
            if len(j) * 2.5 < long {
                across.push((f[j], f[(j + 1) % k]));
            }
        }
    }
    if !across.is_empty() {
        let mut parent: Vec<u32> = (0..nv as u32).collect();
        fn find(p: &mut [u32], mut i: u32) -> u32 {
            while p[i as usize] != i {
                p[i as usize] = p[p[i as usize] as usize];
                i = p[i as usize];
            }
            i
        }
        for &(a, b) in &across {
            let (a, b) = (find(&mut parent, a), find(&mut parent, b));
            parent[a as usize] = b;
        }
        let mut sum: FxHashMap<u32, DVec3> = FxHashMap::default();
        for &(a, b) in &across {
            for i in [a, b] {
                let r = find(&mut parent, i);
                sum.entry(r).or_insert(DVec3::ZERO);
            }
        }
        for i in 0..nv as u32 {
            let r = find(&mut parent, i);
            if let Some(x) = sum.get_mut(&r) {
                *x += n[i as usize];
            }
        }
        for i in 0..nv as u32 {
            let r = find(&mut parent, i);
            if let Some(x) = sum.get(&r) {
                n[i as usize] = x.normalize_or(n[i as usize]);
            }
        }
    }
    let n = n;
    let loops: Vec<&Vec<u32>> = loops.iter().filter(|l| l.len() >= 3).collect();
    // the wall section at every border vertex: its kind's, blended along
    // the loop where the kind changes
    let border: Vec<u32> = (0..nv as u32).filter(|&i| isb[i as usize]).collect();
    let (edge, from) = geodesic(skin, &nb8, &border);
    // the width of the panel at each border vertex: how far the nearest
    // other border is (another loop, or the same one further round than
    // straight across), where that is under 30 mm
    let mut width: FxHashMap<u32, f64> = FxHashMap::default();
    {
        let cell = 0.03;
        let ck = |p: DVec3| {
            (
                (p.x / cell).floor() as i32,
                (p.y / cell).floor() as i32,
                (p.z / cell).floor() as i32,
            )
        };
        // (loop, arc length along it, the loop's length, vertex)
        let mut grid: FxHashMap<(i32, i32, i32), Vec<(usize, f64, f64, u32)>> =
            FxHashMap::default();
        let mut all = Vec::new();
        for (li, lp) in loops.iter().enumerate() {
            let l = lp.len();
            let mut arc = vec![0.0; l + 1];
            for i in 0..l {
                arc[i + 1] =
                    arc[i] + (skin.v[lp[(i + 1) % l] as usize] - skin.v[lp[i] as usize]).length();
            }
            for (i, &vi) in lp.iter().enumerate() {
                let e = (li, arc[i], arc[l], vi);
                grid.entry(ck(skin.v[vi as usize])).or_default().push(e);
                all.push(e);
            }
        }
        for (li, a, total, vi) in all {
            let p = skin.v[vi as usize];
            let k = ck(p);
            let mut best = f64::INFINITY;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(c) = grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) else {
                            continue;
                        };
                        for &(lj, b, _, vj) in c {
                            let dist = (skin.v[vj as usize] - p).length();
                            let round = (a - b).abs().min(total - (a - b).abs());
                            if dist < best && (lj != li || round > 1.8 * dist) {
                                best = dist;
                            }
                        }
                    }
                }
            }
            if best < cell {
                width.insert(vi, best);
            }
        }
        // (along a border the width grows no faster than the border runs:
        // the tip of a spike is as narrow as its sides)
        for lp in &loops {
            let l = lp.len();
            for _ in 0..2 {
                for i in (0..l).chain((0..l).rev()) {
                    for j in [(i + 1) % l, (i + l - 1) % l] {
                        if let Some(&wj) = width.get(&lp[j]) {
                            let x = wj + (skin.v[lp[j] as usize] - skin.v[lp[i] as usize]).length();
                            let e = width.entry(lp[i]).or_insert(f64::INFINITY);
                            *e = e.min(x);
                        }
                    }
                }
            }
        }
    }
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
        // A row s into the panel shortens where the border turns towards
        // the panel and lengthens where it turns away (and the reverse for
        // a row out past the edge): how far a row may go each way before
        // its segments lose over half their length, eased along the loop.
        // A section narrowed to nothing is a plain straight wall.
        let mut lim = [vec![f64::INFINITY; l], vec![f64::INFINITY; l]];
        for i in 0..l {
            let p = |k: usize| skin.v[lp[k % l] as usize];
            let (t0, t1) = (
                (p(i) - p(i + l - 1)).normalize_or_zero(),
                (p(i + 1) - p(i)).normalize_or_zero(),
            );
            let half = 0.5 * t0.dot(t1).clamp(-1.0, 1.0).acos();
            let tan = half.tan().min(50.0);
            if tan > 1e-3 {
                let side = usize::from((t1 - t0).dot(w[i]) < 0.0);
                lim[side][i] = 0.3 * seg[i].min(seg[(i + l - 1) % l]) / tan;
            }
        }
        for lim in &mut lim {
            for _ in 0..2 {
                for i in 0..l {
                    let j = (i + 1) % l;
                    lim[j] = lim[j].min(lim[i] + 0.6 * seg[i]);
                }
                for i in (0..l).rev() {
                    let j = (i + 1) % l;
                    lim[i] = lim[i].min(lim[j] + 0.6 * seg[i]);
                }
            }
        }
        for i in 0..l {
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
                    rows[j].0 += wt[i][k] * r[j].0;
                    rows[j].1 += wt[i][k] * r[j].1;
                }
                inner.0 += wt[i][k] * q.0;
                inner.1 += wt[i][k] * q.1;
            }
            let (mut far_in, mut far_out) = (inner.0.max(0.0), (-inner.0).max(0.0));
            for r in &rows {
                far_in = far_in.max(r.0);
                far_out = far_out.max(-r.0);
            }
            // (and a row goes no further in than 0.3 of the panel's width)
            let wide = width.get(&lp[i]).map_or(f64::INFINITY, |w| 0.3 * w);
            let sc_in = (lim[0][i].min(wide) / far_in.max(1e-9)).min(1.0);
            // (nor out past the edge: the other border may be the far side of a slit)
            let sc_out = (lim[1][i].min(wide) / far_out.max(1e-9)).min(1.0);
            let narrow = |s: f64| if s > 0.0 { s * sc_in } else { s * sc_out };
            for r in &mut rows {
                r.0 = narrow(r.0);
            }
            inner.0 = narrow(inner.0);
            rim.insert(
                lp[i],
                Rim {
                    w: w[i],
                    rows,
                    inner,
                    seg: seg[i].max(seg[(i + l - 1) % l]),
                },
            );
        }
    }
    // the depth field (at a border vertex: the depth of the inner face's rim)
    let glazed: Vec<u32> = border
        .iter()
        .copied()
        .filter(|&i| kind[i as usize] == EdgeKind::Glazed)
        .collect();
    let glass = if glazed.is_empty() {
        vec![f64::INFINITY; nv]
    } else {
        geodesic(skin, &nb8, &glazed).0
    };
    let (lo, hi) = skin.bbox();
    let mut d: Vec<f64> = (0..nv)
        .map(|i| {
            if isb[i] {
                return rim.get(&(i as u32)).map_or(MIN_THICKNESS, |r| r.inner.1);
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
    // the field is sampled once per skin vertex: smooth it over the mesh so
    // a feature's rim running across the grid is a ramp, not stairs
    for _ in 0..2 {
        let prev = d.clone();
        for i in 0..nv {
            if !isb[i] && !nb[i].is_empty() {
                let mean =
                    nb[i].iter().map(|&j| prev[j as usize]).sum::<f64>() / nb[i].len() as f64;
                d[i] = 0.5 * (prev[i] + mean);
            }
        }
    }
    // behind a convex bend an offset surface shrinks, and folds where it is
    // deeper than the bend's radius: no deeper than 0.65 of it there (the
    // estimate is not asked for less than a sheet's thickness)
    let cap = convex_cap(skin, &n, &nb);
    for i in 0..nv {
        d[i] = d[i].min(cap[i].max(SHEET)).max(MIN_THICKNESS);
        // a tip tapering to nothing (a mirror sail) is a blade no deeper
        // than twice its width (or a sheet)
        if let Some(w) = width.get(&(i as u32)) {
            d[i] = d[i].min((2.0 * w).max(SHEET)).max(MIN_THICKNESS);
        }
    }
    let all: Vec<u32> = (0..nv as u32).collect();
    limit_slope(skin, &nb8, &mut d, SLOPE, &all);
    // inner positions; where the offset still squeezes or folds a face the
    // depth is reduced there until it does not
    // A border's inner rim may sit in from the edge (a return flange): the
    // rows behind it make room, the move fading over 60 mm. Then the inner
    // face's cells are evened out along the surface (a thin row of the skin,
    // or one squeezed behind a bend, would be a row of slivers).
    let shift: Vec<DVec3> = (0..nv)
        .map(|i| match rim.get(&from[i]) {
            Some(r) => r.w * (r.inner.0 * (1.0 - edge[i] / 0.06).max(0.0)),
            None => DVec3::ZERO,
        })
        .collect();
    // (no vertex moves over a third of its shortest edge: a small cell stays a cell)
    let reach: Vec<f64> = (0..nv)
        .map(|i| {
            0.35 * nb[i]
                .iter()
                .map(|&j| (skin.v[j as usize] - skin.v[i]).length())
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let place = |d: &[f64], shift: &[DVec3]| -> Vec<DVec3> {
        // (a rim squashed to a fraction of its depth sits in from the edge
        // by that fraction too)
        let q0: Vec<DVec3> = (0..nv)
            .map(|i| {
                let f = rim.get(&from[i]).map_or(1.0, |r| {
                    (d[from[i] as usize] / r.inner.1.max(1e-9)).min(1.0)
                });
                skin.v[i] + shift[i] * f - n[i] * d[i]
            })
            .collect();
        let mut q = q0.clone();
        for _ in 0..EVEN_PASSES {
            let prev = q.clone();
            for i in 0..nv {
                if !isb[i] && !nb[i].is_empty() {
                    let mean =
                        nb[i].iter().map(|&j| prev[j as usize]).sum::<DVec3>() / nb[i].len() as f64;
                    let mv = mean - prev[i];
                    let x = prev[i] + 0.5 * (mv - n[i] * mv.dot(n[i]));
                    q[i] = q0[i] + (x - q0[i]).clamp_length_max(reach[i]);
                }
            }
        }
        q
    };
    // (a face is squeezed when a corner loses its turn or an edge two
    // thirds of its length)
    let squeezed = |q: &[DVec3], f: &[u32]| -> bool {
        let k = f.len();
        let p = |j: usize| skin.v[f[j % k] as usize];
        let g = |j: usize| q[f[j % k] as usize];
        (0..k).any(|j| {
            let no = (p(j + 1) - p(j)).cross(p(j + k - 1) - p(j));
            let ni = (g(j + 1) - g(j)).cross(g(j + k - 1) - g(j));
            ni.dot(no) < 0.2 * ni.length() * no.length()
                || (g(j + 1) - g(j)).length() < 0.34 * (p(j + 1) - p(j)).length()
        })
    };
    // a pinched cell of the skin (an edge under 2.5 mm) that no depth
    // helps is left as it is
    let flat = place(&vec![MIN_THICKNESS; nv], &shift);
    let pinched = |f: &[u32]| {
        (0..f.len()).any(|j| {
            (skin.v[f[(j + 1) % f.len()] as usize] - skin.v[f[j] as usize]).length() < 0.0025
        })
    };
    let live: Vec<&Vec<u32>> = faces
        .iter()
        .filter(|f| !(pinched(f) && squeezed(&flat, f)))
        .collect();
    let mut q = place(&d, &shift);
    for _ in 0..60 {
        let mut bad = vec![false; nv];
        for f in &live {
            if squeezed(&q, f) {
                for &i in f.iter() {
                    bad[i as usize] = true;
                }
            }
        }
        let seeds: Vec<u32> = (0..nv as u32).filter(|&i| bad[i as usize]).collect();
        let mut moved = false;
        for &i in &seeds {
            let x = (d[i as usize] * 0.88).max(MIN_THICKNESS);
            moved |= x < d[i as usize];
            d[i as usize] = x;
        }
        if !moved {
            break;
        }
        limit_slope(skin, &nb8, &mut d, SLOPE, &seeds);
        q = place(&d, &shift);
    }
    // the walls. A section that fits (every row at least a border segment
    // over ROW_ASPECT from the next) is built as drawn; where the wall is
    // too shallow for it (the depth was reduced, the section narrowed at a
    // corner) it is a plain wall of as many evenly spaced rows as fit.
    let mut wall: FxHashMap<u32, (usize, [DVec3; WALL_ROWS])> = FxHashMap::default();
    for lp in &loops {
        let l = lp.len();
        let section = |vi: u32| -> ([DVec3; WALL_ROWS], bool) {
            let r = &rim[&vi];
            let (p, qq) = (skin.v[vi as usize], q[vi as usize]);
            let f = d[vi as usize] / r.inner.1.max(1e-9);
            let x: [DVec3; WALL_ROWS] = std::array::from_fn(|k| {
                p + r.w * (r.rows[k].0 * f.min(1.0)) - n[vi as usize] * (r.rows[k].1 * f)
            });
            let mut least = (x[0] - p).length().min((qq - x[WALL_ROWS - 1]).length());
            for k in 1..WALL_ROWS {
                least = least.min((x[k] - x[k - 1]).length());
            }
            (x, least * ROW_ASPECT >= r.seg)
        };
        let mut sec: Vec<([DVec3; WALL_ROWS], bool)> = lp.iter().map(|&vi| section(vi)).collect();
        // drawn sections ease into a plain stretch over two vertices (and
        // an eased section that no longer fits is plain too)
        let ease = |sec: &[([DVec3; WALL_ROWS], bool)], i: usize| -> f64 {
            let at = |k: isize| {
                f64::from(u8::from(
                    !sec[(i as isize + k).rem_euclid(l as isize) as usize].1,
                ))
            };
            at(0)
                .max(0.6 * at(1).max(at(-1)))
                .max(0.25 * at(2).max(at(-2)))
        };
        let eased = |sec: &[([DVec3; WALL_ROWS], bool)], i: usize| -> (usize, [DVec3; WALL_ROWS]) {
            let vi = lp[i];
            let r = &rim[&vi];
            let (p, qq) = (skin.v[vi as usize], q[vi as usize]);
            let b = ease(sec, i);
            let rows = if b >= 1.0 {
                (((qq - p).length() * ROW_ASPECT / r.seg) as usize)
                    .saturating_sub(1)
                    .min(WALL_ROWS)
            } else {
                WALL_ROWS
            };
            let x: [DVec3; WALL_ROWS] = std::array::from_fn(|k| {
                let straight = p + (qq - p) * ((k + 1) as f64 / (rows + 1) as f64);
                if k >= rows {
                    qq
                } else {
                    sec[i].0[k].lerp(straight, b)
                }
            });
            (rows, x)
        };
        for _ in 0..4 {
            let mut more = Vec::new();
            for i in 0..l {
                if !sec[i].1 {
                    continue;
                }
                let (_, x) = eased(&sec, i);
                let vi = lp[i];
                let (p, qq) = (skin.v[vi as usize], q[vi as usize]);
                let mut least = (x[0] - p).length().min((qq - x[WALL_ROWS - 1]).length());
                for k in 1..WALL_ROWS {
                    least = least.min((x[k] - x[k - 1]).length());
                }
                if least * ROW_ASPECT < rim[&vi].seg {
                    more.push(i);
                }
            }
            if more.is_empty() {
                break;
            }
            for i in more {
                sec[i].1 = false;
            }
        }
        for (i, &vi) in lp.iter().enumerate() {
            wall.insert(vi, eased(&sec, i));
        }
    }
    // assemble: skin, walls, inner face
    let mat = skin
        .qm
        .first()
        .or(skin.tm.first())
        .copied()
        .unwrap_or(Mat::Paint);
    let mut out = skin.clone();
    let nrows: usize = wall.values().map(|w| w.0).sum();
    let inner_of = |i: u32| (nv + nrows) as u32 + i;
    let mut island = island0 + 100;
    let chunk = 1.2;
    for lp in &loops {
        let l = lp.len();
        // row k of loop vertex i: its own vertex, or (a dropped row) the
        // inner face's border vertex
        let mut id = vec![[0u32; WALL_ROWS + 2]; l];
        let mut step = [0.0f64; WALL_ROWS + 1];
        for (i, &vi) in lp.iter().enumerate() {
            let (rows, x) = &wall[&vi];
            id[i][0] = vi;
            let mut prev = skin.v[vi as usize];
            for k in 0..WALL_ROWS {
                id[i][k + 1] = if k < *rows {
                    out.v.push(x[k]);
                    out.v.len() as u32 - 1
                } else {
                    inner_of(vi)
                };
                step[k] += (x[k] - prev).length() / l as f64;
                prev = x[k];
            }
            id[i][WALL_ROWS + 1] = inner_of(vi);
            step[WALL_ROWS] += (q[vi as usize] - prev).length() / l as f64;
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
                let (a0, b0, a1, b1) = (id[i][k], id[j][k], id[i][k + 1], id[j][k + 1]);
                let uv = [
                    DVec2::new(ub, vv[k]),
                    DVec2::new(ua, vv[k]),
                    DVec2::new(ua, vv[k + 1]),
                    DVec2::new(ub, vv[k + 1]),
                ];
                match (a0 == a1, b0 == b1) {
                    (false, false) => {
                        out.q.push([b0, a0, a1, b1]);
                        out.quv.push(uv);
                        out.qi.push(island + cf as i32);
                        out.qm.push(mat);
                    }
                    (true, true) => {}
                    (ta, _) => {
                        // a row ends here: a triangle
                        let (f, u) = if ta {
                            ([b0, a0, b1], [uv[0], uv[1], uv[3]])
                        } else {
                            ([b0, a0, a1], [uv[0], uv[1], uv[2]])
                        };
                        out.t.push(f);
                        out.tuv.push(u);
                        out.ti.push(island + cf as i32);
                        out.tm.push(mat);
                    }
                }
            }
        }
        island += py_floordiv(s[l], chunk) as i32 + 1;
    }
    let mut surf = vec![Surf::Outer; nv];
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
    let mut back: Vec<DVec3> = Vec::with_capacity(m.v.len());
    {
        let out = &mut back;
        if drawn {
            let n = m.vertex_normals();
            let dir = n.iter().copied().sum::<DVec3>().normalize_or_zero();
            // one move for every vertex (a varying one would cross where the
            // surface runs along the draw): 1.4 th, so the lens is th thick
            // where it faces 45 degrees off the draw
            out.extend(m.v.iter().map(|p| *p - dir * (1.4 * th)));
        } else {
            let n = smooth_normals(m, 8);
            let cap = convex_cap(m, &n, &neighbours(m));
            // thinner (down to a third) behind a bend tighter than the slab is thick
            out.extend((0..m.v.len()).map(|i| m.v[i] - n[i] * th.min(cap[i]).max(th / 3.0)));
        }
    }
    slab_to(m, &back)
}

/// A single-sided surface closed with a given back (one point per vertex)
/// and a wall round every border.
pub fn slab_to(m: &Mesh, back: &[DVec3]) -> Mesh {
    let nv = m.v.len() as u32;
    let mut out = m.clone();
    out.v.extend_from_slice(back);
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

/// Drop the faces that keep a single-sided mesh from being closed as a
/// slab: those round a vertex where two sheets of faces meet at a point
/// (more than two border edges there), then any island of fewer than
/// `least` faces.
pub fn single_sheet(m: &mut Mesh, least: usize) {
    for _ in 0..12 {
        let mut dir: FxHashSet<(u32, u32)> = FxHashSet::default();
        for f in m.faces() {
            for i in 0..f.len() {
                dir.insert((f[i], f[(i + 1) % f.len()]));
            }
        }
        let mut open = vec![0u8; m.v.len()];
        for &(a, b) in &dir {
            if !dir.contains(&(b, a)) {
                open[a as usize] += 1;
                open[b as usize] += 1;
            }
        }
        if open.iter().all(|c| *c <= 2) {
            break;
        }
        drop_faces(m, |f| f.iter().any(|&i| open[i as usize] > 2));
    }
    let comp = components(m);
    let mut size: FxHashMap<u32, usize> = FxHashMap::default();
    for f in m.faces() {
        *size.entry(comp[f[0] as usize]).or_insert(0) += 1;
    }
    drop_faces(m, |f| size[&comp[f[0] as usize]] < least);
}

fn drop_faces(m: &mut Mesh, gone: impl Fn(&[u32]) -> bool) {
    let kq: Vec<bool> = m.q.iter().map(|f| !gone(f)).collect();
    crate::mesh::retain_by(&mut m.q, &kq);
    crate::mesh::retain_by(&mut m.qm, &kq);
    crate::mesh::retain_by(&mut m.quv, &kq);
    crate::mesh::retain_by(&mut m.qi, &kq);
    let kt: Vec<bool> = m.t.iter().map(|f| !gone(f)).collect();
    crate::mesh::retain_by(&mut m.t, &kt);
    crate::mesh::retain_by(&mut m.tm, &kt);
    crate::mesh::retain_by(&mut m.tuv, &kt);
    crate::mesh::retain_by(&mut m.ti, &kt);
}

/// A door's trim card: a moulding `th` thick facing the cabin, `depth`
/// below the door's skin, with an armrest standing out of it, a pull cup
/// let into the armrest and a map pocket along the bottom. Built on the
/// door skin's cells, two rings in from its edges; where the skin bends too
/// tightly for a face that far behind it (a rear door's dogleg round the
/// wheel arch) the card stops short.
pub fn door_card(skin: &Mesh, depth: f64, th: f64, mat: Mat) -> Mesh {
    let nv = skin.v.len();
    let nb = neighbours(skin);
    let mut ring = vec![false; nv];
    for lp in boundary_loops(skin) {
        for i in lp {
            ring[i as usize] = true;
        }
    }
    for _ in 0..2 {
        let prev = ring.clone();
        for i in 0..nv {
            if !prev[i] && nb[i].iter().any(|&j| prev[j as usize]) {
                ring[i] = true;
            }
        }
    }
    let n = smooth_normals(skin, 6);
    let (lo, hi) = skin.bbox();
    let size = hi - lo;
    let (ya, zc) = (lo.y + 0.47 * size.y, lo.z + 0.60 * size.z);
    let mut d: Vec<f64> = skin
        .v
        .iter()
        .map(|p| {
            let along = smoothstep(lo.z + 0.10 * size.z, lo.z + 0.18 * size.z, p.z)
                * (1.0 - smoothstep(hi.z - 0.18 * size.z, hi.z - 0.10 * size.z, p.z));
            let arm = 0.045 * (1.0 - smoothstep(0.040, 0.066, (p.y - ya).abs())) * along;
            let cup = 0.030
                * (1.0 - smoothstep(0.010, 0.040, (p.y - ya).abs()))
                * (1.0 - smoothstep(0.050, 0.090, (p.z - zc).abs()));
            let pocket =
                0.016 * (1.0 - smoothstep(lo.y + 0.17 * size.y, lo.y + 0.24 * size.y, p.y)) * along;
            depth + arm - cup + pocket
        })
        .collect();
    for _ in 0..2 {
        let prev = d.clone();
        for i in 0..nv {
            if !nb[i].is_empty() {
                let mean =
                    nb[i].iter().map(|&j| prev[j as usize]).sum::<f64>() / nb[i].len() as f64;
                d[i] = 0.5 * (prev[i] + mean);
            }
        }
    }
    // no card where its face would fold behind a bend of the skin
    let cap = convex_cap(skin, &n, &nb);
    let fold: Vec<bool> = (0..nv).map(|i| cap[i] < d[i]).collect();
    let keep: Vec<bool> = (0..nv)
        .map(|i| !ring[i] && !fold[i] && !nb[i].iter().any(|&j| fold[j as usize]))
        .collect();
    let mut remap = vec![u32::MAX; nv];
    let mut m = Mesh::new();
    let mut back = Vec::new();
    for i in 0..nv {
        if keep[i] {
            remap[i] = m.v.len() as u32;
            m.v.push(skin.v[i] - n[i] * d[i]);
            back.push(skin.v[i] - n[i] * (d[i] - th));
        }
    }
    for (k, f) in skin.q.iter().enumerate() {
        if f.iter().all(|&i| keep[i as usize]) {
            m.q.push(f.map(|i| remap[i as usize]));
            m.qm.push(mat);
            m.quv.push(skin.quv[k]);
            m.qi.push(-1);
        }
    }
    for (k, f) in skin.t.iter().enumerate() {
        if f.iter().all(|&i| keep[i as usize]) {
            m.t.push(f.map(|i| remap[i as usize]));
            m.tm.push(mat);
            m.tuv.push(skin.tuv[k]);
            m.ti.push(-1);
        }
    }
    // it faces the cabin
    m.flip();
    single_sheet(&mut m, 300);
    slab_to(&m, &back)
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

/// Painted triangles of a part past the art spec's aspect limit (longest
/// edge squared over twice the area, `check.py`'s definition), counted by
/// what they are: [skin, edge walls, inner face].
pub fn aspect_over(m: &Mesh, limit: f64) -> [usize; 3] {
    let mut n = [0; 3];
    for (k, _, _) in slivers(m, limit) {
        n[k] += 1;
    }
    n
}

/// The triangles [`aspect_over`] counts: (0 skin, 1 wall, 2 inner face;
/// centroid; aspect).
pub fn slivers(m: &Mesh, limit: f64) -> Vec<(usize, DVec3, f64)> {
    let mut out = Vec::new();
    for (f, _, mat) in crate::export::tri_corners(m) {
        if mat != Mat::Paint {
            continue;
        }
        let p = f.map(|i| m.v[i as usize]);
        let l2 = (p[1] - p[0])
            .length_squared()
            .max((p[2] - p[1]).length_squared())
            .max((p[0] - p[2]).length_squared());
        let a2 = (p[1] - p[0]).cross(p[2] - p[0]).length().max(2e-14);
        if l2 > limit * a2 {
            let s = f.map(|i| m.surf_of(i as usize));
            let k = if s.iter().all(|x| *x == Surf::Outer) {
                0
            } else if s.iter().all(|x| *x == Surf::Inner) {
                2
            } else {
                1
            };
            out.push((k, (p[0] + p[1] + p[2]) / 3.0, l2 / a2));
        }
    }
    out
}

/// Faces of a part that cut through each other: edges of every `stride`-th
/// triangle against the triangles near them (sharing no vertex). Returns the
/// crossings' points.
pub fn self_intersections(m: &Mesh, stride: usize) -> Vec<DVec3> {
    crossings(m, stride).into_iter().map(|c| c.3).collect()
}

/// Where a part's drawn skin is folded: where it cuts through itself, and
/// where one of its edges passes through a wall hanging from its own border
/// (a cell of the skin turned over the border: a spike or a slit a couple
/// of millimetres wide at a tight corner of the sheet).
pub fn skin_folds(m: &Mesh) -> Vec<DVec3> {
    let mut out = self_intersections(&m.outer_only(), 1);
    let o = |i: u32| m.surf_of(i as usize) == Surf::Outer;
    out.extend(
        crossings(m, 1)
            .into_iter()
            .filter(|(a, b, t, _)| o(*a) && o(*b) && t.iter().filter(|i| o(**i)).count() == 2)
            .map(|c| c.3),
    );
    out
}

/// Faces of a solid cutting through each other away from the folds of its
/// drawn skin ([`skin_folds`]; 30 mm).
pub fn new_crossings(m: &Mesh) -> Vec<DVec3> {
    let folds = skin_folds(m);
    self_intersections(m, 1)
        .into_iter()
        .filter(|p| !folds.iter().any(|s| (*s - *p).length() < 0.03))
        .collect()
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

/// Pairs of parts [`pair_crossings`] is worth checking for: a pane or a lens
/// against a panel or another pane, a door card or the body shell (its
/// B-pillar, its jambs) against a door.
pub fn neighbours_to_check(a: &str, b: &str) -> bool {
    let glassy = |n: &str| n.starts_with("glass") || n.ends_with("_lens");
    let panel = |n: &str| crate::uvpack::is_panel(n);
    let door = |n: &str| n.starts_with("door_F") || n.starts_with("door_R");
    let inner = |n: &str| n.starts_with("door_trim") || n == "body_shell";
    (glassy(a) && (panel(b) || glassy(b)))
        || (glassy(b) && panel(a))
        || (inner(a) && door(b))
        || (inner(b) && door(a))
}

/// Edges of either part passing through a face of the other (two parts
/// cutting through each other: a pane through its rebate).
pub fn pair_crossings(a: &Mesh, b: &Mesh) -> Vec<DVec3> {
    let mut hits = Vec::new();
    for (a, b) in [(a, b), (b, a)] {
        let tris = crate::export::split_triangles(b);
        let cell = 0.04;
        let ck = |x: f64| (x / cell).floor() as i32;
        let mut grid: FxHashMap<(i32, i32, i32), Vec<u32>> = FxHashMap::default();
        for (ti, f) in tris.iter().enumerate() {
            let p = f.map(|i| b.v[i as usize]);
            let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
            for x in ck(lo.x)..=ck(hi.x) {
                for y in ck(lo.y)..=ck(hi.y) {
                    for z in ck(lo.z)..=ck(hi.z) {
                        grid.entry((x, y, z)).or_default().push(ti as u32);
                    }
                }
            }
        }
        let mut seen: FxHashSet<(u32, u32, u32)> = FxHashSet::default();
        for f in crate::export::split_triangles(a) {
            for e in 0..3 {
                let (ia, ib) = (f[e], f[(e + 1) % 3]);
                let (p, q) = (a.v[ia as usize], a.v[ib as usize]);
                let (lo, hi) = (p.min(q), p.max(q));
                for x in ck(lo.x)..=ck(hi.x) {
                    for y in ck(lo.y)..=ck(hi.y) {
                        for z in ck(lo.z)..=ck(hi.z) {
                            let Some(c) = grid.get(&(x, y, z)) else {
                                continue;
                            };
                            for &tj in c {
                                if !seen.insert((ia.min(ib), ia.max(ib), tj)) {
                                    continue;
                                }
                                let g = tris[tj as usize].map(|i| b.v[i as usize]);
                                if let Some(x) = segment_triangle(p, q, g[0], g[1], g[2]) {
                                    hits.push(x);
                                }
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
