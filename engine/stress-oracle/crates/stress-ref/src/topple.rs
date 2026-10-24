//! Topple: a narrow reinforced-concrete high-rise beside a street, brought down by
//! demolition charges (The Finals' street-building topple, Teardown's high-rise topple).
//!
//! [`topple_tower`] is [`wrecking::rc_frame`] at 2 x 1 bays and 10 storeys (12.5 x 6.5 m
//! in plan, 35.5 m tall: slender enough to tip) on pad footings: the Wrecking Yard
//! frame's sections and storeys, with shear walls across both short ends from the second
//! storey up (they carry the street side's weight across to the back once its columns
//! are gone) and a back wall along the ground storey's back line on strip footings (the
//! tower then stands on the whole back line, not on three column feet). A grade-level
//! street of slab tiles (supports) runs along its long street side (-y). Its two charges
//! (`Structure::charges`, the pack's `defaults.charges`):
//!
//! 1. [`STREET_SIDE`] (at once): every bond of the three ground-storey columns on the
//!    street side, to their footings and to the joints above. The blast throws the cut
//!    columns out into the street ([`ToppleLayout::KICK`]), as a demolition's kicker
//!    charges do: a rigid column cut loose would otherwise stand where it was and carry
//!    the storeys above as before.
//! 2. [`BACK_ROW`] ([`ToppleLayout::HINGE_DELAY`] later, once the columns are out of the
//!    way): the bonds of the back line (its three ground-storey columns and the back wall)
//!    to their footings. The tower then stands on the back line alone, its weight over
//!    the empty street side, and leans out over the street about it.
//!
//! Every chunk sits at or above grade; the footings and the street's tiles are supports.

use crate::builders::auto_bonds;
use crate::scene::{ChunkDesc, Support};
use crate::wrecking::{self, between_free, block, footing, joint_of, Aim, Charge, FrameLayout, Structure, WallRun, BALLS};

/// The tower's and the street's dimensions (m); the frame's own are [`FrameLayout`]'s.
pub struct ToppleLayout;

impl ToppleLayout {
    /// Bays along x (the long side, facing the street) and along y.
    pub const BAYS: [usize; 2] = [2, 1];
    pub const STOREYS: usize = 10;
    /// Between the street-side footings' edge and the street's near edge.
    pub const SIDEWALK: f64 = 2.0;
    /// The street: across (along -y), along the tower (x, centred on it), its depth and
    /// its tiles' plan side.
    pub const STREET_WIDTH: f64 = 20.0;
    pub const STREET_LENGTH: f64 = 24.0;
    pub const STREET_DEPTH: f64 = 0.3;
    pub const TILE: f64 = 4.0;
    /// Grid lines: (bays + 1) along x times (bays + 1) along y.
    pub const LINES: usize = (Self::BAYS[0] + 1) * (Self::BAYS[1] + 1);
    /// Chunks per storey of the frame: a column and a joint on every grid line, the beam
    /// halves (x beams on each y line, y beams on each x line, two a span) and the slab
    /// quarters.
    pub const PER_STOREY: usize = 2 * Self::LINES + 2 * ((Self::BAYS[1] + 1) * Self::BAYS[0] + (Self::BAYS[0] + 1) * Self::BAYS[1]) + 4 * Self::BAYS[0] * Self::BAYS[1];
    /// Wall halves: the two end walls' (one bay each) in every storey but the ground's,
    /// the back wall's (both bays) in the ground storey; and the back wall's strip
    /// footings.
    pub const WALLS: usize = 2 * 2 * (Self::STOREYS - 1) + 2 * Self::BAYS[0];
    pub const STRIPS: usize = 2 * Self::BAYS[0];
    pub const TILES: usize = ((Self::STREET_LENGTH / Self::TILE) as usize) * ((Self::STREET_WIDTH / Self::TILE) as usize);
    /// The storeys, a pad footing on every grid line, the strips and the street's tiles.
    pub const CHUNKS: usize = Self::STOREYS * Self::PER_STOREY + Self::WALLS + Self::LINES + Self::STRIPS + Self::TILES;
    /// The speed the street-side charges throw their columns out at (m/s, Z-up): out over
    /// the street and a little up.
    pub const KICK: [f64; 3] = [0.0, -12.0, 1.0];
    /// When the back line is cut from its footings, after the street-side columns (s).
    pub const HINGE_DELAY: f64 = 0.25;
}

/// The charges' names, in firing order.
pub const STREET_SIDE: &str = "street-side columns";
pub const BACK_ROW: &str = "back-row footings";

/// The tower's wall runs ([`WallRun`]): the end walls on the x lines 0 and 2 from the
/// second storey up, the back wall on y line 1 in the ground storey.
pub fn walls() -> Vec<WallRun> {
    type L = ToppleLayout;
    let end = |line: usize| WallRun { along_x: false, line, bays: 0..L::BAYS[1], storeys: 1..L::STOREYS };
    vec![end(0), end(L::BAYS[0]), WallRun { along_x: true, line: L::BAYS[1], bays: 0..L::BAYS[0], storeys: 0..1 }]
}

/// A 10-storey, 2 x 1 bay reinforced-concrete frame with end shear walls, beside a 20 m
/// street of 4 m slab tiles, with the charges that topple it across the street (the
/// module's notes). 420 chunks.
pub fn topple_tower() -> Structure {
    type L = ToppleLayout;
    type F = FrameLayout;
    let mut chunks = wrecking::rc_frame(L::BAYS, L::STOREYS, &walls());
    // The street, along the tower's street side (-y), at grade.
    let half_y = 0.5 * L::BAYS[1] as f64 * F::BAY;
    let near = -half_y - 0.5 * F::FOOTING - L::SIDEWALK;
    let (nx, ny) = ((L::STREET_LENGTH / L::TILE) as usize, (L::STREET_WIDTH / L::TILE) as usize);
    for j in 0..ny {
        for i in 0..nx {
            let x0 = -0.5 * L::STREET_LENGTH + i as f64 * L::TILE;
            let y1 = near - j as f64 * L::TILE;
            chunks.push(footing(block([x0, y1 - L::TILE, 0.0], [x0 + L::TILE, y1, L::STREET_DEPTH], "footing-anchor", "street", "street".into())));
        }
    }
    let bonds = between_free(&chunks, auto_bonds(&chunks, joint_of));
    // The ground storey's columns on the street-side grid line (y line 0); the back line's
    // columns and wall.
    let on_line = |k: usize, y: f64| (chunks[k].center[1] - y).abs() < 1e-9;
    let street: Vec<usize> = (0..chunks.len()).filter(|&k| ground(&chunks[k], "column") && on_line(k, -half_y)).collect();
    let back: Vec<usize> = (0..chunks.len()).filter(|&k| (ground(&chunks[k], "column") || ground(&chunks[k], "wall")) && on_line(k, half_y)).collect();
    let touching = |set: &[usize], other: &dyn Fn(&ChunkDesc) -> bool| -> Vec<usize> {
        (0..bonds.len()).filter(|&i| (set.contains(&bonds[i].a) && other(&chunks[bonds[i].b])) || (set.contains(&bonds[i].b) && other(&chunks[bonds[i].a]))).collect()
    };
    let charges = vec![
        Charge { name: STREET_SIDE.into(), delay: 0.0, bonds: touching(&street, &|_| true), debris: street.clone(), kick: L::KICK },
        Charge { name: BACK_ROW.into(), delay: L::HINGE_DELAY, bonds: touching(&back, &|c| c.support == Support::Fixed), debris: Vec::new(), kick: [0.0; 3] },
    ];
    let top = F::FOOTING_DEPTH + L::STOREYS as f64 * F::STOREY;
    Structure {
        key: "topple_tower",
        title: "Topple: high-rise beside a street",
        chunks,
        bonds,
        // A ball from behind at the back face, half way up: toward the street.
        aim: Aim { target: [0.0, half_y + 0.5 * F::COLUMN, 0.5 * top], direction: [0.0, -1.0, 0.0] },
        ball: BALLS[1],
        charges,
    }
}

/// A chunk of type `kind` (`column`, `wall`, ...) in the ground storey.
pub fn ground(c: &ChunkDesc, kind: &str) -> bool {
    c.groups.first().map(String::as_str) == Some(kind) && c.groups.get(1).map(String::as_str) == Some("storey-1")
}
