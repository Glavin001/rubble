//! Buildings for the vehicle and aircraft beats, built as the Wrecking Yard's structures
//! are (`wrecking::Structure`: Z-up boxes at or above grade, footings `Support::Fixed`,
//! `groups[0]` the node type and `groups[1]` its `storey-N`) and written as scene packs by
//! `wrecking::to_pack`:
//!
//! * [`storefront_row`] (Ram Raid): a terrace of five two-storey shops: brick party walls,
//!   a concrete first-floor slab and roof, glass shopfronts between brick piers under a
//!   concrete fascia, and a brick upper facade with windows under concrete lintels.
//! * [`apartment_block`] (Fly-through): a seven-storey reinforced-concrete frame, 4 x 2
//!   bays of 6 m, its perimeter bays filled with masonry panels round a window each.
//! * [`glass_tower`] (Fly-through): an eleven-storey tower of concrete floor slabs on a
//!   walled core and perimeter columns, wrapped in a glass curtain wall of panes on steel
//!   mullions and transoms.
//!
//! Every chunk is an axis-aligned box, so the bonds are `auto_bonds` between touching
//! faces (the joint material `wrecking::joint_of`), less those between two supports
//! (`between_free`) and those a glazed wall does not have ([`glazing_joints`]).
//!
//! Brick walls are laid in running bond (`masonry`): units of `Bond::unit` m in courses of
//! `Bond::course` m, every other course shifted half a unit, the joints on one grid for the
//! whole building, so the courses of every wall stagger. The units are bigger than real
//! bricks (as the brick tower's are): a chunk stands for a few bricks.

use crate::builders::auto_bonds;
use crate::scene::{BondDesc, ChunkDesc};
use crate::wrecking::{between_free, block, footing, joint_of, Aim, Structure};

const RC: &str = "reinforced-concrete";
const FOOTING: &str = "footing-anchor";

/// A box spanning `u` along a wall (x or y), `v` across it and `z` up.
fn wall_box(along_x: bool, u: (f64, f64), v: (f64, f64), z: (f64, f64), material: &str, kind: &str, group: &str) -> ChunkDesc {
    if along_x {
        block([u.0, v.0, z.0], [u.1, v.1, z.1], material, kind, group.into())
    } else {
        block([v.0, u.0, z.0], [v.1, u.1, z.1], material, kind, group.into())
    }
}

/// A box spanning `x`, `y` and `z`.
fn cuboid(x: (f64, f64), y: (f64, f64), z: (f64, f64), material: &str, kind: &str, group: &str) -> ChunkDesc {
    block([x.0, y.0, z.0], [x.1, y.1, z.1], material, kind, group.into())
}

/// `lo..hi` cut at `cuts` (those strictly inside), in order.
fn pieces(lo: f64, hi: f64, cuts: &[f64]) -> Vec<(f64, f64)> {
    let mut at: Vec<f64> = cuts.iter().copied().filter(|&c| c > lo + 1e-9 && c < hi - 1e-9).collect();
    at.sort_by(f64::total_cmp);
    let mut edges = vec![lo];
    edges.extend(at);
    edges.push(hi);
    edges.windows(2).map(|w| (w[0], w[1])).collect()
}

/// An opening in a masonry wall: `u0..u1` along it, courses `c0..c1` (the wall's own,
/// from 0), glazed with `panes` panes side by side (none: left open) and, with a
/// `lintel`, spanned by a concrete lintel in course `c1` bearing that far (m) on each side.
#[derive(Clone, Copy, Debug)]
pub struct Opening {
    pub u0: f64,
    pub u1: f64,
    pub c0: usize,
    pub c1: usize,
    pub panes: usize,
    pub lintel: Option<f64>,
}

/// A masonry wall: along x or y, `u` along it and `v` across it, `courses` courses from
/// `z0`, of type `kind` in `group`.
#[derive(Clone, Copy, Debug)]
pub struct Wall<'a> {
    pub along_x: bool,
    pub u: (f64, f64),
    pub v: (f64, f64),
    pub z0: f64,
    pub courses: usize,
    pub kind: &'a str,
    pub group: &'a str,
}

/// The running bond's unit and course (m).
#[derive(Clone, Copy, Debug)]
pub struct Bond {
    pub unit: f64,
    pub course: f64,
    /// Glass thickness of the panes in the wall's openings (centred in the wall).
    pub glass: f64,
}

/// The units of one course over `lo..hi`: cut on the bond's grid (shifted half a unit on
/// odd courses, counted from z = 0), no unit shorter than a quarter unit unless the run
/// itself is (a short end merges into its neighbour).
fn course_units(lo: f64, hi: f64, shift: f64, unit: f64) -> Vec<(f64, f64)> {
    let (a, b) = (lo + 0.25 * unit, hi - 0.25 * unit);
    let mut cuts = Vec::new();
    let mut k = ((a - shift) / unit).floor() as i64;
    loop {
        let x = shift + k as f64 * unit;
        if x >= b {
            break;
        }
        if x > a {
            cuts.push(x);
        }
        k += 1;
    }
    pieces(lo, hi, &cuts)
}

/// Lay `wall` in running bond with `openings`: bricks round them, their panes and their
/// lintels.
pub fn masonry(chunks: &mut Vec<ChunkDesc>, wall: &Wall, openings: &[Opening], bond: Bond) {
    let (v0, v1) = wall.v;
    let vm = 0.5 * (v0 + v1);
    let z = |c: usize| wall.z0 + c as f64 * bond.course;
    for c in 0..wall.courses {
        let (z0, z1) = (z(c), z(c + 1));
        // What is not brick in this course: the openings it crosses and the lintels over them.
        let mut holes: Vec<(f64, f64)> = Vec::new();
        for o in openings {
            if (o.c0..o.c1).contains(&c) {
                holes.push((o.u0, o.u1));
            }
            if let (true, Some(bear)) = (o.c1 == c, o.lintel) {
                holes.push((o.u0 - bear, o.u1 + bear));
                chunks.push(wall_box(wall.along_x, (o.u0 - bear, o.u1 + bear), wall.v, (z0, z1), RC, "lintel", wall.group));
            }
        }
        holes.sort_by(|a, b| a.0.total_cmp(&b.0));
        let parity = ((z0 / bond.course).round() as i64).rem_euclid(2);
        let shift = 0.5 * bond.unit * parity as f64;
        let mut at = wall.u.0;
        for &(h0, h1) in holes.iter().chain(std::iter::once(&(wall.u.1, wall.u.1))) {
            if h0 > at + 1e-9 {
                for u in course_units(at, h0, shift, bond.unit) {
                    chunks.push(wall_box(wall.along_x, u, wall.v, (z0, z1), "brick", wall.kind, wall.group));
                }
            }
            at = at.max(h1);
        }
    }
    for o in openings.iter().filter(|o| o.panes > 0) {
        let w = (o.u1 - o.u0) / o.panes as f64;
        for k in 0..o.panes {
            let u = (o.u0 + k as f64 * w, o.u0 + (k + 1) as f64 * w);
            chunks.push(wall_box(wall.along_x, u, (vm - 0.5 * bond.glass, vm + 0.5 * bond.glass), (z(o.c0), z(o.c1)), "glass", "glass", wall.group));
        }
    }
}

/// The joints glazing does not have, left out of the touching faces' bonds, so that glass
/// carries no load of the building's:
/// * a pane is bonded at its sill, where it stands on setting blocks, and, in a curtain
///   wall, at its sides to the mullions and corner posts (structural silicone glazing);
/// * not at its head (a clearance under the frame above), nor at its sides in a masonry
///   or concrete opening or against the next pane (a frame or glazing bar with a movement
///   clearance, sealed but not structural), so the masonry round it, as it settles, loads
///   nothing into it.
///
/// (A curtain wall's panes have no sill to stand on: each storey's hangs from its slab
/// edge over an open stack joint, `TowerLayout::STACK_JOINT`.) A bond is vertical when its
/// normal is (the boxes' faces are axis-aligned).
pub fn glazing_joints(chunks: &[ChunkDesc], mut bonds: Vec<BondDesc>) -> Vec<BondDesc> {
    fn kind(c: &ChunkDesc) -> &str {
        c.groups.first().map(String::as_str).unwrap_or("")
    }
    bonds.retain(|b| {
        let (a, c) = (&chunks[b.a], &chunks[b.b]);
        match (a.material == "glass", c.material == "glass") {
            (false, false) => true,
            (true, true) => false,
            (true, false) | (false, true) => {
                let (pane, other) = if a.material == "glass" { (a, c) } else { (c, a) };
                if b.normal[2].abs() > 0.5 {
                    // Its sill: the pane above what it stands on.
                    pane.center[2] > other.center[2]
                } else {
                    matches!(kind(other), "mullion" | "post")
                }
            }
        }
    });
    bonds
}

/// The bonds of a building of boxes: touching faces, not between two supports, not those
/// glazing lacks.
fn building_bonds(chunks: &[ChunkDesc]) -> Vec<BondDesc> {
    glazing_joints(chunks, between_free(chunks, auto_bonds(chunks, joint_of)))
}

fn storey(s: usize) -> String {
    format!("storey-{s}")
}

/// The storefront row's dimensions (m).
pub struct StorefrontLayout;

impl StorefrontLayout {
    pub const SHOPS: usize = 5;
    /// Party wall centre to centre.
    pub const WIDTH: f64 = 5.5;
    /// Front face to back face.
    pub const DEPTH: f64 = 7.0;
    pub const WALL: f64 = 0.3;
    pub const BOND: Bond = Bond { unit: 1.0, course: 0.5, glass: 0.02 };
    /// Ground-storey courses of the party and back walls (3.5 m); the front's piers and
    /// stallrisers have one fewer under the fascia.
    pub const GROUND_COURSES: usize = 7;
    pub const UPPER_COURSES: usize = 6;
    pub const SLAB: f64 = 0.2;
    pub const FOOTING_DEPTH: f64 = 0.5;
    /// Brick piers at the party walls (the end piers too) and each shop's glazed door.
    pub const PIER: f64 = 0.6;
    pub const DOOR: f64 = 1.1;
    /// Upper windows: two a shop, this wide, either side of its centre by `WINDOW_AT`, in
    /// courses 2..5 under a lintel bearing `BEARING` each side.
    pub const WINDOW: f64 = 1.0;
    pub const WINDOW_AT: f64 = 1.3;
    pub const BEARING: f64 = 0.25;

    /// Outside half-length (the gables' outer faces).
    pub fn half_length() -> f64 {
        0.5 * Self::SHOPS as f64 * Self::WIDTH + 0.5 * Self::WALL
    }

    /// The party walls' centres, gables included (k = 0..=SHOPS).
    pub fn party(k: usize) -> f64 {
        (k as f64 - 0.5 * Self::SHOPS as f64) * Self::WIDTH
    }

    /// The pier at party wall k along the street.
    pub fn pier(k: usize) -> (f64, f64) {
        let (x, h, p) = (Self::party(k), 0.5 * Self::WALL, Self::PIER);
        match k {
            0 => (x - h, x - h + p),
            k if k == Self::SHOPS => (x + h - p, x + h),
            _ => (x - 0.5 * p, x + 0.5 * p),
        }
    }

    /// Shop i's glazed door and shop window along the street (doors alternate sides).
    pub fn shopfront(i: usize) -> ((f64, f64), (f64, f64)) {
        let (p, q) = (Self::pier(i).1, Self::pier(i + 1).0);
        if i % 2 == 0 {
            ((p, p + Self::DOOR), (p + Self::DOOR, q))
        } else {
            ((q - Self::DOOR, q), (p, q - Self::DOOR))
        }
    }

    /// Shop i's two upper windows along the street.
    pub fn windows(i: usize) -> [(f64, f64); 2] {
        let c = 0.5 * (Self::party(i) + Self::party(i + 1));
        let h = 0.5 * Self::WINDOW;
        [(c - Self::WINDOW_AT - h, c - Self::WINDOW_AT + h), (c + Self::WINDOW_AT - h, c + Self::WINDOW_AT + h)]
    }
}

/// A terrace of five two-storey shops, 5.5 m wide and 7 m deep (27.8 x 7 m, 7.4 m to the
/// roof), on concrete strip footings. Brick throughout in running bond: 0.3 m party walls
/// and gables, a solid back wall, the front's brick piers and stallrisers on the ground
/// storey and its upper facade with two windows a shop under concrete lintels. Each
/// shopfront is a glazed door and a two-pane shop window between the piers under a
/// concrete fascia (0.3 x 0.5 m). The 0.2 m concrete first-floor slab and roof bear on
/// the walls. Panes are 20 mm glass. The ball comes in from the street at the middle
/// shop's window, 1.25 m up.
pub fn storefront_row() -> Structure {
    type L = StorefrontLayout;
    let (x, d, t) = (L::half_length(), 0.5 * L::DEPTH, L::WALL);
    let bond = L::BOND;
    let base = L::FOOTING_DEPTH;
    let ground_top = base + L::GROUND_COURSES as f64 * bond.course;
    let (upper0, upper_top) = (ground_top + L::SLAB, ground_top + L::SLAB + L::UPPER_COURSES as f64 * bond.course);
    let front = (-d, -d + t);
    let back = (d - t, d);
    let party_v = |k: usize| (L::party(k) - 0.5 * t, L::party(k) + 0.5 * t);
    let inner = (-d + t, d - t);
    let mut chunks = Vec::new();
    // Strip footings under the front and back walls (a shop each) and each party wall.
    let mut shop_cuts: Vec<f64> = (1..L::SHOPS).map(L::party).collect();
    for v in [front, back] {
        for u in pieces(-x, x, &shop_cuts) {
            chunks.push(footing(wall_box(true, u, v, (0.0, base), FOOTING, "footing", "footing")));
        }
    }
    for k in 0..=L::SHOPS {
        chunks.push(footing(wall_box(false, inner, party_v(k), (0.0, base), FOOTING, "footing", "footing")));
    }
    let (g1, g2) = (storey(1), storey(2));
    // Ground storey: the front's piers, stallrisers, doors and shop windows, and the
    // fascia over them; the back wall and the party walls.
    let front_courses = L::GROUND_COURSES - 1;
    let mut shopfronts = Vec::new();
    for i in 0..L::SHOPS {
        let (door, window) = L::shopfront(i);
        shopfronts.push(Opening { u0: door.0, u1: door.1, c0: 0, c1: front_courses, panes: 1, lintel: None });
        shopfronts.push(Opening { u0: window.0, u1: window.1, c0: 1, c1: front_courses, panes: 2, lintel: None });
    }
    masonry(&mut chunks, &Wall { along_x: true, u: (-x, x), v: front, z0: base, courses: front_courses, kind: "front-wall", group: &g1 }, &shopfronts, bond);
    let fascia0 = base + front_courses as f64 * bond.course;
    let centres: Vec<f64> = (0..L::SHOPS).map(|i| 0.5 * (L::party(i) + L::party(i + 1))).collect();
    shop_cuts.extend(&centres);
    for u in pieces(-x, x, &shop_cuts) {
        chunks.push(wall_box(true, u, front, (fascia0, ground_top), RC, "fascia", &g1));
    }
    masonry(&mut chunks, &Wall { along_x: true, u: (-x, x), v: back, z0: base, courses: L::GROUND_COURSES, kind: "back-wall", group: &g1 }, &[], bond);
    for k in 0..=L::SHOPS {
        masonry(&mut chunks, &Wall { along_x: false, u: inner, v: party_v(k), z0: base, courses: L::GROUND_COURSES, kind: "party-wall", group: &g1 }, &[], bond);
    }
    // The slabs: a piece from each party wall's centre to each shop's centre, in three
    // strips front to back.
    let strips: Vec<f64> = (1..3).map(|k| -d + k as f64 * L::DEPTH / 3.0).collect();
    let slab = |chunks: &mut Vec<ChunkDesc>, z: (f64, f64), kind: &str, group: &str| {
        for u in pieces(-x, x, &shop_cuts) {
            for v in pieces(-d, d, &strips) {
                chunks.push(cuboid(u, v, z, RC, kind, group));
            }
        }
    };
    slab(&mut chunks, (ground_top, upper0), "slab", &g1);
    // Upper storey: the facade's windows, the back wall, the party walls; the roof.
    let windows: Vec<Opening> = (0..L::SHOPS).flat_map(|i| L::windows(i).map(|(u0, u1)| Opening { u0, u1, c0: 2, c1: 5, panes: 1, lintel: Some(L::BEARING) })).collect();
    masonry(&mut chunks, &Wall { along_x: true, u: (-x, x), v: front, z0: upper0, courses: L::UPPER_COURSES, kind: "front-wall", group: &g2 }, &windows, bond);
    masonry(&mut chunks, &Wall { along_x: true, u: (-x, x), v: back, z0: upper0, courses: L::UPPER_COURSES, kind: "back-wall", group: &g2 }, &[], bond);
    for k in 0..=L::SHOPS {
        masonry(&mut chunks, &Wall { along_x: false, u: inner, v: party_v(k), z0: upper0, courses: L::UPPER_COURSES, kind: "party-wall", group: &g2 }, &[], bond);
    }
    slab(&mut chunks, (upper_top, upper_top + L::SLAB), "roof", &g2);
    let bonds = building_bonds(&chunks);
    let (_, window) = L::shopfront(L::SHOPS / 2);
    Structure {
        key: "rr_storefront_row",
        title: "Storefront row",
        chunks,
        bonds,
        aim: Aim { target: [0.5 * (window.0 + window.1), -d, base + 1.25], direction: [0.0, 1.0, 0.0] },
        ball: *crate::wrecking::ball("car2t").expect("the vehicle stand-in"),
    }
}

/// The apartment block's dimensions (m).
pub struct ApartmentLayout;

impl ApartmentLayout {
    pub const BAYS_X: usize = 4;
    pub const BAYS_Y: usize = 2;
    pub const BAY: f64 = 6.0;
    pub const STOREYS: usize = 7;
    pub const STOREY: f64 = 3.0;
    pub const COLUMN: f64 = 0.45;
    pub const BEAM_DEPTH: f64 = 0.5;
    /// As wide as the columns (the slabs start at the beams' and the joints' faces).
    pub const BEAM_WIDTH: f64 = 0.45;
    pub const SLAB: f64 = 0.2;
    /// Masonry infill panels, centred on the perimeter grid lines.
    pub const INFILL: f64 = 0.25;
    /// Each infill bay's window: width, sill height and head height above the floor.
    pub const WINDOW: f64 = 2.0;
    pub const SILL: f64 = 0.9;
    pub const HEAD: f64 = 2.2;
    pub const GLASS: f64 = 0.02;
    /// The ground storey's glazed entrance: front bay `ENTRANCE` (no sill under it).
    pub const ENTRANCE: usize = 1;
    pub const FOOTING: f64 = 1.0;
    pub const FOOTING_DEPTH: f64 = 0.5;
    /// Grid lines along x and y.
    pub const LINES: usize = (Self::BAYS_X + 1) * (Self::BAYS_Y + 1);
    /// Beam halves: two a span along each of the grid's lines.
    pub const BEAMS: usize = 2 * ((Self::BAYS_Y + 1) * Self::BAYS_X + (Self::BAYS_X + 1) * Self::BAYS_Y);
    pub const PERIMETER_BAYS: usize = 2 * (Self::BAYS_X + Self::BAYS_Y);
    /// Chunks per storey: columns, joints, beam halves, slab quarters, and per perimeter
    /// bay two sill halves, two jambs, two lintel halves and the pane.
    pub const PER_STOREY: usize = 2 * Self::LINES + Self::BEAMS + 4 * Self::BAYS_X * Self::BAYS_Y + 7 * Self::PERIMETER_BAYS;
    /// The storeys (the entrance has no sill), the pad footings and a strip footing under
    /// each perimeter bay.
    pub const CHUNKS: usize = Self::STOREYS * Self::PER_STOREY - 2 + Self::LINES + Self::PERIMETER_BAYS;

    pub fn line_x(i: usize) -> f64 {
        (i as f64 - 0.5 * Self::BAYS_X as f64) * Self::BAY
    }

    pub fn line_y(j: usize) -> f64 {
        (j as f64 - 0.5 * Self::BAYS_Y as f64) * Self::BAY
    }
}

/// A seven-storey reinforced-concrete apartment block, 4 x 2 bays of 6 m (24.45 x 12.45
/// m, 21.5 m tall), 3 m storeys: a 0.45 m column on every grid line, a beam-column joint
/// above it, 0.45 x 0.5 m beams in two chunks a span and a 0.2 m slab in four chunks a bay,
/// on 1 m pad footings. Every perimeter bay is filled under its beam with 0.25 m masonry
/// panels round a 2 x 1.3 m window (sill halves and jambs, mortar to the frame) under a
/// concrete lintel the bay's width (two halves, under the beam), on strip footings at the
/// ground; one front bay of the ground storey is a glazed entrance. The ball comes in at
/// the front's middle column at the fourth storey.
pub fn apartment_block() -> Structure {
    type L = ApartmentLayout;
    let (nx, ny) = (L::BAYS_X + 1, L::BAYS_Y + 1);
    let (hc, hw, hi) = (0.5 * L::COLUMN, 0.5 * L::BEAM_WIDTH, 0.5 * L::INFILL);
    let base = L::FOOTING_DEPTH;
    let top = |s: usize| base + (s + 1) as f64 * L::STOREY;
    let mut chunks = Vec::with_capacity(L::CHUNKS);
    let hf = 0.5 * L::FOOTING;
    for j in 0..ny {
        for i in 0..nx {
            let (x, y) = (L::line_x(i), L::line_y(j));
            chunks.push(footing(cuboid((x - hf, x + hf), (y - hf, y + hf), (0.0, base), FOOTING, "footing", "footing")));
        }
    }
    // The perimeter bays: (along x, the span's two grid lines along it, the line it is on).
    let mut bays: Vec<(bool, f64, f64, f64, bool)> = Vec::new();
    for &j in &[0, L::BAYS_Y] {
        for i in 0..L::BAYS_X {
            bays.push((true, L::line_x(i), L::line_x(i + 1), L::line_y(j), j == 0 && i == L::ENTRANCE));
        }
    }
    for &i in &[0, L::BAYS_X] {
        for j in 0..L::BAYS_Y {
            bays.push((false, L::line_y(j), L::line_y(j + 1), L::line_x(i), false));
        }
    }
    for &(along_x, a, b, line, _) in &bays {
        chunks.push(footing(wall_box(along_x, (a + hf, b - hf), (line - hw, line + hw), (0.0, base), FOOTING, "footing", "footing")));
    }
    for s in 0..L::STOREYS {
        let (z1, group) = (top(s), storey(s + 1));
        let z0 = if s == 0 { base } else { top(s - 1) };
        let zb = z1 - L::BEAM_DEPTH;
        for j in 0..ny {
            for i in 0..nx {
                let (x, y) = (L::line_x(i), L::line_y(j));
                chunks.push(cuboid((x - hc, x + hc), (y - hc, y + hc), (z0, zb), RC, "column", &group));
                chunks.push(cuboid((x - hc, x + hc), (y - hc, y + hc), (zb, z1), RC, "joint", &group));
            }
        }
        // Beams along x (on each y line) and along y (on each x line), two halves a span.
        for j in 0..ny {
            for i in 0..L::BAYS_X {
                let (a, b) = (L::line_x(i) + hc, L::line_x(i + 1) - hc);
                for u in pieces(a, b, &[0.5 * (a + b)]) {
                    chunks.push(wall_box(true, u, (L::line_y(j) - hw, L::line_y(j) + hw), (zb, z1), RC, "beam", &group));
                }
            }
        }
        for i in 0..nx {
            for j in 0..L::BAYS_Y {
                let (a, b) = (L::line_y(j) + hc, L::line_y(j + 1) - hc);
                for u in pieces(a, b, &[0.5 * (a + b)]) {
                    chunks.push(wall_box(false, u, (L::line_x(i) - hw, L::line_x(i) + hw), (zb, z1), RC, "beam", &group));
                }
            }
        }
        // Slabs between the beams, four quarters a bay, flush with the beams' tops.
        for j in 0..L::BAYS_Y {
            for i in 0..L::BAYS_X {
                let (x0, x1, y0, y1) = (L::line_x(i) + hw, L::line_x(i + 1) - hw, L::line_y(j) + hw, L::line_y(j + 1) - hw);
                for u in pieces(x0, x1, &[0.5 * (x0 + x1)]) {
                    for v in pieces(y0, y1, &[0.5 * (y0 + y1)]) {
                        chunks.push(cuboid(u, v, (z1 - L::SLAB, z1), RC, "slab", &group));
                    }
                }
            }
        }
        // Infill: between the columns' faces, from the floor (beam top or footing) to the
        // beam above.
        for &(along_x, a, b, line, entrance) in &bays {
            let (a, b) = (a + hc, b - hc);
            let m = 0.5 * (a + b);
            let (w0, w1) = (m - 0.5 * L::WINDOW, m + 0.5 * L::WINDOW);
            let v = (line - hi, line + hi);
            let (sill, head) = (z0 + L::SILL, z0 + L::HEAD);
            let entrance = entrance && s == 0;
            let bottom = if entrance { z0 } else { sill };
            if !entrance {
                for u in [(a, m), (m, b)] {
                    chunks.push(wall_box(along_x, u, v, (z0, sill), "brick", "infill", &group));
                }
            }
            chunks.push(wall_box(along_x, (a, w0), v, (bottom, head), "brick", "infill", &group));
            chunks.push(wall_box(along_x, (w1, b), v, (bottom, head), "brick", "infill", &group));
            for u in [(a, m), (m, b)] {
                chunks.push(wall_box(along_x, u, v, (head, zb), RC, "lintel", &group));
            }
            chunks.push(wall_box(along_x, (w0, w1), (line - 0.5 * L::GLASS, line + 0.5 * L::GLASS), (bottom, head), "glass", "glass", &group));
        }
    }
    let bonds = building_bonds(&chunks);
    let mid = L::STOREYS / 2;
    Structure {
        key: "ft_apartment_block",
        title: "Apartment block",
        chunks,
        bonds,
        aim: Aim { target: [L::line_x(L::BAYS_X / 2), L::line_y(0) - hc, top(mid) - 0.5 * L::STOREY], direction: [0.0, 1.0, 0.0] },
        ball: *crate::wrecking::ball("jet").expect("the aircraft stand-in"),
    }
}

/// The glass tower's dimensions (m).
pub struct TowerLayout;

impl TowerLayout {
    pub const STOREYS: usize = 11;
    pub const STOREY: f64 = 3.6;
    /// Half the slab's plan side, and the slab grid's lines (each side's last cell is
    /// the edge strip the perimeter columns stand in).
    pub const HALF: f64 = 6.0;
    pub const GRID: [f64; 7] = [-6.0, -4.25, -2.5, 0.0, 2.5, 4.25, 6.0];
    /// The core: outer half-side and wall.
    pub const CORE: f64 = 2.5;
    pub const CORE_WALL: f64 = 0.3;
    pub const SLAB: f64 = 0.25;
    /// Perimeter columns at the corners and mid-sides, this far out.
    pub const COLUMN: f64 = 0.45;
    pub const COLUMN_AT: f64 = 5.5;
    /// Curtain wall: bays a side, mullion width and depth (outside the slab edge), panes.
    pub const BAYS: usize = 6;
    pub const MULLION: f64 = 0.08;
    pub const MULLION_DEPTH: f64 = 0.16;
    pub const GLASS: f64 = 0.025;
    /// The stack joint: the open gap between one storey's curtain wall and the next one's
    /// (each hangs from its own slab edge; real stack joints leave 15-25 mm for movement).
    pub const STACK_JOINT: f64 = 0.02;
    pub const FOOTING: f64 = 1.0;
    pub const FOOTING_DEPTH: f64 = 0.5;
    /// Chunks per storey: 32 slab cells, 8 core wall halves, 8 columns, per side
    /// (BAYS - 1) mullions and BAYS panes and transoms, 4 corner posts.
    pub const PER_STOREY: usize = 32 + 8 + 8 + 4 * (3 * Self::BAYS - 1) + 4;
    /// The storeys, 8 pad footings and 4 core strip footings.
    pub const CHUNKS: usize = Self::STOREYS * Self::PER_STOREY + 12;

    /// The columns' plan positions.
    pub fn columns() -> [(f64, f64); 8] {
        let c = Self::COLUMN_AT;
        [(-c, -c), (0.0, -c), (c, -c), (-c, 0.0), (c, 0.0), (-c, c), (0.0, c), (c, c)]
    }
}

/// An eleven-storey tower, 12 m square and 40 m tall: 0.25 m concrete floor slabs (32
/// cells a floor) on a 5 m square walled core (0.3 m walls, the lift shaft open) and eight
/// 0.45 m perimeter columns, wrapped in a curtain wall outside the slab edges: 25 mm
/// glass panes in six 2 m bays a side between steel mullions (80 x 160 mm), on a steel
/// transom at each floor's edge, with corner posts. The curtain wall is hung storey by
/// storey: each storey's mullions and transom are anchored to its slab's edge and its
/// panes held by the mullions beside them ([`glazing_joints`]), with a 20 mm stack joint
/// open under it (`STACK_JOINT`), so no storey's curtain wall carries another's, or the
/// building. The ball comes in at the front's middle mullion at mid-height (storey 6).
pub fn glass_tower() -> Structure {
    type L = TowerLayout;
    let (h, k, w) = (L::HALF, L::CORE, L::CORE_WALL);
    let (md, mw, hc) = (L::MULLION_DEPTH, 0.5 * L::MULLION, 0.5 * L::COLUMN);
    let base = L::FOOTING_DEPTH;
    let top = |s: usize| base + (s + 1) as f64 * L::STOREY;
    let mut chunks = Vec::with_capacity(L::CHUNKS);
    let hf = 0.5 * L::FOOTING;
    for (x, y) in L::columns() {
        chunks.push(footing(cuboid((x - hf, x + hf), (y - hf, y + hf), (0.0, base), FOOTING, "footing", "footing")));
    }
    // Core walls: the two along x run the core's full side, the two along y between them.
    let core_walls = |z: (f64, f64), group: &str, material: &str, kind: &str, halves: bool| -> Vec<ChunkDesc> {
        let mut out = Vec::new();
        let split = |lo: f64, hi: f64| if halves { pieces(lo, hi, &[0.0]) } else { vec![(lo, hi)] };
        for v in [(-k, -k + w), (k - w, k)] {
            for u in split(-k, k) {
                out.push(wall_box(true, u, v, z, material, kind, group));
            }
        }
        for v in [(-k, -k + w), (k - w, k)] {
            for u in split(-k + w, k - w) {
                out.push(wall_box(false, u, v, z, material, kind, group));
            }
        }
        out
    };
    for c in core_walls((0.0, base), "footing", FOOTING, "footing", false) {
        chunks.push(footing(c));
    }
    let edge = h + md;
    // A side's bays along it: the mullions' centres and the panes' (and transoms') spans.
    let step = 2.0 * h / L::BAYS as f64;
    let mullions: Vec<f64> = (1..L::BAYS).map(|i| -h + i as f64 * step).collect();
    let spans: Vec<(f64, f64)> = (0..L::BAYS).map(|i| (if i == 0 { -h } else { mullions[i - 1] + mw }, if i + 1 == L::BAYS { h } else { mullions[i] - mw })).collect();
    for s in 0..L::STOREYS {
        let (z1, group) = (top(s), storey(s + 1));
        let z0 = if s == 0 { base } else { top(s - 1) };
        let zs = z1 - L::SLAB;
        chunks.extend(core_walls((z0, z1), &group, RC, "core", true));
        for (i, x) in L::GRID.windows(2).enumerate() {
            for (j, y) in L::GRID.windows(2).enumerate() {
                if (2..4).contains(&i) && (2..4).contains(&j) {
                    continue;
                }
                chunks.push(cuboid((x[0], x[1]), (y[0], y[1]), (zs, z1), RC, "slab", &group));
            }
        }
        for (x, y) in L::columns() {
            chunks.push(cuboid((x - hc, x + hc), (y - hc, y + hc), (z0, zs), RC, "column", &group));
        }
        // Curtain wall, side by side: along x at y = -h and +h, along y at x = -h and +h,
        // the mullion depth outside the slab edge, from the stack joint over the storey
        // below's transom to the top of this storey's slab.
        let zc = z0 + L::STACK_JOINT;
        for (along_x, v) in [(true, (-edge, -h)), (true, (h, edge)), (false, (-edge, -h)), (false, (h, edge))] {
            let vm = 0.5 * (v.0 + v.1);
            for &m in &mullions {
                chunks.push(wall_box(along_x, (m - mw, m + mw), v, (zc, z1), "steel", "mullion", &group));
            }
            for &u in &spans {
                chunks.push(wall_box(along_x, u, v, (zs, z1), "steel", "transom", &group));
                chunks.push(wall_box(along_x, u, (vm - 0.5 * L::GLASS, vm + 0.5 * L::GLASS), (zc, zs), "glass", "glass", &group));
            }
        }
        for x in [(-edge, -h), (h, edge)] {
            for y in [(-edge, -h), (h, edge)] {
                chunks.push(cuboid(x, y, (zc, z1), "steel", "post", &group));
            }
        }
    }
    let bonds = building_bonds(&chunks);
    let mid = L::STOREYS / 2;
    Structure {
        key: "ft_glass_tower",
        title: "Glass tower",
        chunks,
        bonds,
        aim: Aim { target: [0.0, -edge, top(mid) - 0.5 * L::STOREY], direction: [0.0, 1.0, 0.0] },
        ball: *crate::wrecking::ball("jet").expect("the aircraft stand-in"),
    }
}
