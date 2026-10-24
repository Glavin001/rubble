//! Skybridge: an enclosed steel walkway between two reinforced-concrete towers, for the
//! beat where a span drops at one end, hangs from the other, then lets go (The Finals'
//! skybridge failures; `engine/docs/demo-scenarios.md` S2 and S6). Built as the Wrecking
//! Yard's structures are (`wrecking`: boxes, `auto_bonds`, written as a scene pack).
//!
//! * Two towers, each [`wrecking::rc_frame`] at 2 x 2 bays and 7 storeys (12.5 m square,
//!   25 m tall), 20 m apart face to face along x, on pad footings.
//! * The walkway joins them at the fifth storey's floor (14.5 m up). It is a 3.6 m wide,
//!   19.2 m long Vierendeel truss of S355 steel: two bottom chords, two top chords and
//!   posts at 2.4 m panel points, a 0.2 m concrete deck between the bottom chords, a 40 mm
//!   steel roof plate between the top chords and 25 mm glass panes between the posts
//!   (bonded at their sills and to the posts, `buildings::glazing_joints`). Its four
//!   sections (`span-1` to `span-4`, west to east, two panels each) are typed as the
//!   viaduct's spans are (`deck`), so `viaduct::span_probes` follows them.
//! * Its connections are the weak joints. At each end a concrete ledge is cast onto the
//!   tower's facade beam, flush with the floor, and each bottom chord sits on a steel
//!   bearing plate on it. The bearings' bonds to the ledges (0.32 m^2 of concrete joint
//!   each) are the walkway's only bonds to the towers: a seismic joint
//!   ([`SkybridgeLayout::JOINT`]) separates its ends from the facades. Beyond each bearing
//!   two keepers on the ledge, one on each side of the chord and clear of it, stop the
//!   bearing sliding off the ledge.
//! * A plaza between the towers, under the walkway: concrete slabs bonded to a sub-base
//!   of supports, for the secondary impact.
//!
//! The charges (`Structure::charges`, the pack's `defaults.charges`):
//! 1. [`EAST`] (at once): every bond of the east end's ledges, bearings and keepers. The
//!    blast throws them down and away from the facade, so the walkway's east end has
//!    nothing under it.
//! 2. [`WEST`] ([`SkybridgeLayout::TEAR_DELAY`] later): the west bearings' bonds to their
//!    ledges.
//!
//! What then holds the walkway is contact alone: its west bearings on the west ledge,
//! held from sliding off by the keepers. It swings down about them, east end first, until
//! its bottom chords come down on the ledge's outer edge (about 12 degrees, the east end
//! 4 m down). The walkway then levers against the keepers and the ledge; the ledge tears
//! out of the facade beam, and the walkway drops onto the plaza.
//!
//! Under `blast_parity` the east charge alone is the trigger: the quasi-static solve sees
//! the walkway cantilevered off its west bearings and breaks their bonds within a few
//! frames, before the west charge (which then finds nothing to cut). Under `threshold`,
//! which breaks bonds on contact force alone, a cantilever loads no bond, so the west
//! charge is what lets the walkway go.
//!
//! Every chunk sits at or above grade; the towers' footings and the plaza's sub-base are
//! supports.

use crate::builders::auto_bonds;
use crate::buildings::glazing_joints;
use crate::scene::ChunkDesc;
use crate::wrecking::{self, between_free, block, footing, joint_of, Aim, Charge, FrameLayout, Structure, BALLS};

/// The skybridge's dimensions (m). x runs from the west tower to the east one, y across
/// the walkway, z up.
pub struct SkybridgeLayout;

impl SkybridgeLayout {
    /// Each tower: bays (x, y) and storeys of the Wrecking Yard frame.
    pub const BAYS: [usize; 2] = [2, 2];
    pub const STOREYS: usize = 7;
    /// Between the towers' facing facades.
    pub const GAP: f64 = 20.0;
    /// The storey whose floor the walkway joins (1-based): the fifth, 14.5 m up.
    pub const STOREY: usize = 5;
    /// The walkway's centre line across (y): the middle of the facade's second bay, clear
    /// of the facade's columns.
    pub const CENTRE_Y: f64 = 3.0;
    pub const WIDTH: f64 = 3.6;
    /// The seismic joint between each end of the walkway and the facade. Wide enough that
    /// the walkway swings down about either end's bearings clear of the far facade.
    pub const JOINT: f64 = 0.4;
    pub const PANELS: usize = 8;
    /// Sections (`span-N`), of `PANELS / SECTIONS` panels each.
    pub const SECTIONS: usize = 4;
    /// Steel members: chord width (y) and depths, post side (x).
    pub const CHORD: f64 = 0.25;
    pub const BOTTOM_CHORD: f64 = 0.4;
    pub const TOP_CHORD: f64 = 0.25;
    pub const POST: f64 = 0.2;
    /// The posts' (and panes') clear height between the chords.
    pub const CLEAR: f64 = 2.4;
    /// Concrete deck depth, flush with the bottom chords' tops; roof plate, flush with the
    /// top chords'; deck and roof chunks across.
    pub const DECK: f64 = 0.2;
    pub const ROOF: f64 = 0.04;
    pub const ACROSS: usize = 2;
    pub const GLASS: f64 = 0.025;
    /// A bearing plate under each chord end: length along the walkway, width (across it)
    /// and depth.
    pub const BEARING: [f64; 3] = [0.4, 0.8, 0.1];
    /// Each end's ledge, cast onto the facade beam: projection from the facade, half its
    /// length across (y, centred on the walkway's line) and depth (the beam's, its top
    /// the floor's).
    pub const LEDGE: [f64; 3] = [1.25, 2.2, FrameLayout::BEAM_DEPTH];
    /// The keepers beyond each bearing, on the ledge: the gap to the bearing, the
    /// clearance to the chord's side, and how far they stand above the bearing's top.
    pub const KEEPER: [f64; 3] = [0.05, 0.02, 0.05];
    /// The plaza: from x = -`PLAZA_HALF_X`, tiles of `TILE` square, `PLAZA_X` along x and
    /// `PLAZA_Y` along y (centred on the walkway's line), `PLAZA_DEPTH` thick on a sub-base
    /// `PLAZA_BASE` thick.
    pub const TILE: f64 = 3.0;
    pub const PLAZA_X: usize = 6;
    pub const PLAZA_Y: usize = 4;
    pub const PLAZA_HALF_X: f64 = 0.5 * Self::PLAZA_X as f64 * Self::TILE;
    pub const PLAZA_DEPTH: f64 = 0.25;
    pub const PLAZA_BASE: f64 = 0.2;
    /// The speed the east charge throws its ledges, bearings and keepers at (m/s, Z-up):
    /// down and away from the east facade.
    pub const KICK: [f64; 3] = [-2.0, 0.0, -3.0];
    /// When the west bearings are cut, after the east connection (s): about when
    /// `blast_parity`'s stress solve breaks them.
    pub const TEAR_DELAY: f64 = 0.2;

    /// Grid lines (x, y) of a tower: (bays + 1)^2.
    pub const LINES: usize = (Self::BAYS[0] + 1) * (Self::BAYS[1] + 1);
    /// A tower storey: columns and joints, beam halves, slab quarters.
    pub const PER_STOREY: usize = 2 * Self::LINES + 2 * ((Self::BAYS[1] + 1) * Self::BAYS[0] + (Self::BAYS[0] + 1) * Self::BAYS[1]) + 4 * Self::BAYS[0] * Self::BAYS[1];
    pub const TOWER: usize = Self::STOREYS * Self::PER_STOREY + Self::LINES;
    /// The walkway: four chords of `PANELS` chunks, `PANELS + 1` posts and `PANELS` panes
    /// a side, `ACROSS` deck and roof chunks a panel.
    pub const WALKWAY: usize = 4 * Self::PANELS + 2 * (Self::PANELS + 1) + 2 * Self::PANELS + 2 * Self::ACROSS * Self::PANELS;
    /// Each end: two ledge chunks, a bearing and two keepers a chord.
    pub const CONNECTIONS: usize = 2 * (2 + 2 * 3);
    pub const PLAZA: usize = 2 * Self::PLAZA_X * Self::PLAZA_Y;
    pub const CHUNKS: usize = 2 * Self::TOWER + Self::WALKWAY + Self::CONNECTIONS + Self::PLAZA;

    /// A tower's half side to its facade's outer face (the column's).
    pub fn tower_half() -> f64 {
        0.5 * Self::BAYS[0] as f64 * FrameLayout::BAY + 0.5 * FrameLayout::COLUMN
    }
    /// The towers' centres along x (west, east).
    pub fn tower_x() -> [f64; 2] {
        let c = 0.5 * Self::GAP + Self::tower_half();
        [-c, c]
    }
    /// The floor the walkway joins: the top of storey `STOREY - 1`.
    pub fn floor() -> f64 {
        FrameLayout::FOOTING_DEPTH + (Self::STOREY - 1) as f64 * FrameLayout::STOREY
    }
    /// The ledges' tops: the floor's.
    pub fn ledge_top() -> f64 {
        Self::floor()
    }
    /// The bottom chords' soffit, on the bearings.
    pub fn chord_bottom() -> f64 {
        Self::ledge_top() + Self::BEARING[2]
    }
    pub fn post_bottom() -> f64 {
        Self::chord_bottom() + Self::BOTTOM_CHORD
    }
    pub fn top_chord_bottom() -> f64 {
        Self::post_bottom() + Self::CLEAR
    }
    pub fn roof_top() -> f64 {
        Self::top_chord_bottom() + Self::TOP_CHORD
    }
    /// The walkway's ends (x).
    pub fn ends() -> (f64, f64) {
        (-0.5 * Self::GAP + Self::JOINT, 0.5 * Self::GAP - Self::JOINT)
    }
    pub fn panel() -> f64 {
        let (x0, x1) = Self::ends();
        (x1 - x0) / Self::PANELS as f64
    }
    /// The chords' centre lines (y).
    pub fn chord_y() -> [f64; 2] {
        let off = 0.5 * Self::WIDTH - 0.5 * Self::CHORD;
        [Self::CENTRE_Y - off, Self::CENTRE_Y + off]
    }
}

/// The charges' names, in firing order.
pub const EAST: &str = "east connection";
pub const WEST: &str = "west connection";

/// Two seven-storey concrete towers joined at the fifth floor by an enclosed steel
/// walkway on bearings, over a plaza, with the charges that drop it (module doc). 992
/// chunks, 1,728 bonds, about 2,200 t (the walkway 119 t). The ball comes in along +y at
/// the east ledge.
pub fn skybridge() -> Structure {
    type L = SkybridgeLayout;
    let rc = "reinforced-concrete";
    let steel = "steel";
    let mut chunks: Vec<ChunkDesc> = Vec::with_capacity(L::CHUNKS);
    for x in L::tower_x() {
        for mut c in wrecking::rc_frame(L::BAYS, L::STOREYS, &[]) {
            c.center[0] += x;
            chunks.push(c);
        }
    }
    let (x0, x1) = L::ends();
    let p = L::panel();
    let px = |i: usize| x0 + i as f64 * p;
    let section = |i: usize| format!("span-{}", 1 + i * L::SECTIONS / L::PANELS);
    let hc = 0.5 * L::CHORD;
    let (yin0, yin1) = (L::chord_y()[0] + hc, L::chord_y()[1] - hc);
    let (zb, zp, zt, zr) = (L::chord_bottom(), L::post_bottom(), L::top_chord_bottom(), L::roof_top());
    // The walkway, panel by panel: chords, deck and roof; the posts at the panel points;
    // the panes between them.
    for i in 0..L::PANELS {
        let (a, b, g) = (px(i), px(i + 1), section(i));
        for yc in L::chord_y() {
            chunks.push(block([a, yc - hc, zb], [b, yc + hc, zp], steel, "chord", g.clone()));
            chunks.push(block([a, yc - hc, zt], [b, yc + hc, zr], steel, "chord", g.clone()));
        }
        let w = (yin1 - yin0) / L::ACROSS as f64;
        for k in 0..L::ACROSS {
            let (c, d) = (yin0 + k as f64 * w, yin0 + (k + 1) as f64 * w);
            chunks.push(block([a, c, zp - L::DECK], [b, d, zp], rc, "deck", g.clone()));
            chunks.push(block([a, c, zr - L::ROOF], [b, d, zr], steel, "roof", g.clone()));
        }
    }
    let hp = 0.5 * L::POST;
    let post_x = |i: usize| -> (f64, f64) {
        if i == 0 {
            (x0, x0 + L::POST)
        } else if i == L::PANELS {
            (x1 - L::POST, x1)
        } else {
            (px(i) - hp, px(i) + hp)
        }
    };
    for yc in L::chord_y() {
        for i in 0..=L::PANELS {
            let (a, b) = post_x(i);
            chunks.push(block([a, yc - hc, zp], [b, yc + hc, zt], steel, "post", section(i.min(L::PANELS - 1))));
        }
        for i in 0..L::PANELS {
            let (a, b) = (post_x(i).1, post_x(i + 1).0);
            let hg = 0.5 * L::GLASS;
            chunks.push(block([a, yc - hg, zp], [b, yc + hg, zt], "glass", "glass", section(i)));
        }
    }
    // The connections, at each end (the facade beam's face at x = -GAP / 2 or GAP / 2): a
    // ledge cast onto the facade beam (two chunks, split as the beam is, in the storey
    // under the walkway's floor as the beam is), a bearing plate under each chord's end on
    // it (in the walkway's end section) and, a little beyond each bearing, a keeper on each
    // side of the chord that stops the bearing sliding off.
    let zc = L::ledge_top();
    let beam_storey = format!("storey-{}", L::STOREY - 1);
    let span = |a: f64, b: f64| (a.min(b), a.max(b));
    for (end, face, dir, group) in [("west", -0.5 * L::GAP, 1.0, section(0)), ("east", 0.5 * L::GAP, -1.0, section(L::PANELS - 1))] {
        let kind = |k: &str| format!("{k}-{end}");
        let (l0, l1) = span(face, face + dir * L::LEDGE[0]);
        for (y0, y1) in [(L::CENTRE_Y - L::LEDGE[1], L::CENTRE_Y), (L::CENTRE_Y, L::CENTRE_Y + L::LEDGE[1])] {
            chunks.push(block([l0, y0, zc - L::LEDGE[2]], [l1, y1, zc], rc, &kind("ledge"), beam_storey.clone()));
        }
        let (b0, b1) = span(face + dir * L::JOINT, face + dir * (L::JOINT + L::BEARING[0]));
        let (k0, k1) = span(face + dir * (L::JOINT + L::BEARING[0] + L::KEEPER[0]), face + dir * L::LEDGE[0]);
        let hb = 0.5 * L::BEARING[1];
        for yc in L::chord_y() {
            chunks.push(block([b0, yc - hb, zc], [b1, yc + hb, zb], steel, &kind("bearing"), group.clone()));
            let inner = hc + L::KEEPER[1];
            for (y0, y1) in [(yc - hb, yc - inner), (yc + inner, yc + hb)] {
                chunks.push(block([k0, y0, zc], [k1, y1, zb + L::KEEPER[2]], rc, &kind("keeper"), beam_storey.clone()));
            }
        }
    }
    // The plaza between the towers: slabs on a sub-base (the supports).
    for j in 0..L::PLAZA_Y {
        for i in 0..L::PLAZA_X {
            let (xa, ya) = (-L::PLAZA_HALF_X + i as f64 * L::TILE, L::CENTRE_Y - 0.5 * L::PLAZA_Y as f64 * L::TILE + j as f64 * L::TILE);
            let (xb, yb) = (xa + L::TILE, ya + L::TILE);
            chunks.push(footing(block([xa, ya, 0.0], [xb, yb, L::PLAZA_BASE], "footing-anchor", "subbase", "plaza".into())));
            chunks.push(block([xa, ya, L::PLAZA_BASE], [xb, yb, L::PLAZA_BASE + L::PLAZA_DEPTH], rc, "plaza", "plaza".into()));
        }
    }
    debug_assert_eq!(chunks.len(), L::CHUNKS);
    let bonds = glazing_joints(&chunks, between_free(&chunks, auto_bonds(&chunks, joint_of)));
    let of_kind = |k: &str| -> Vec<usize> { (0..chunks.len()).filter(|&i| chunks[i].groups[0] == k).collect() };
    let east: Vec<usize> = ["ledge-east", "bearing-east", "keeper-east"].iter().flat_map(|k| of_kind(k)).collect();
    let (west_bearings, west_ledges) = (of_kind("bearing-west"), of_kind("ledge-west"));
    let touching = |set: &[usize], other: &dyn Fn(usize) -> bool| -> Vec<usize> { (0..bonds.len()).filter(|&i| (set.contains(&bonds[i].a) && other(bonds[i].b)) || (set.contains(&bonds[i].b) && other(bonds[i].a))).collect() };
    let charges = vec![
        Charge { name: EAST.into(), delay: 0.0, bonds: touching(&east, &|_| true), debris: east.clone(), kick: L::KICK },
        Charge { name: WEST.into(), delay: L::TEAR_DELAY, bonds: touching(&west_bearings, &|k| west_ledges.contains(&k)), debris: Vec::new(), kick: [0.0; 3] },
    ];
    let south = L::CENTRE_Y - L::LEDGE[1];
    Structure {
        key: "sb_skybridge",
        title: "Skybridge between two towers",
        chunks,
        bonds,
        aim: Aim { target: [0.5 * L::GAP - 0.5 * L::LEDGE[0], south, zc - 0.5 * L::LEDGE[2]], direction: [0.0, 1.0, 0.0] },
        ball: BALLS[0],
        charges,
    }
}

/// The walkway's chunks (its sections', `span-N`: the bearings among them), by index.
pub fn walkway(chunks: &[ChunkDesc]) -> Vec<usize> {
    (0..chunks.len()).filter(|&k| chunks[k].groups.get(1).is_some_and(|g| g.starts_with("span-"))).collect()
}
