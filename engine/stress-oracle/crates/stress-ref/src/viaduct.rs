//! Bridge Out: a reinforced-concrete highway viaduct for the pier-hit beat (Red Faction's
//! and The Finals' bridge failures, BeamNG's collapsing-bridge pileup), built as the
//! Wrecking Yard's structures are (`wrecking`: boxes, `auto_bonds`, written as a scene
//! pack).
//!
//! Four 16 m spans (support line to line) of a 9 m wide, 0.9 m deep deck, its top 10 m
//! above grade, on three hammerhead piers and two abutments. The deck is continuous: every
//! span is bonded to the next over each pier.
//! * Each pier is a 1.8 m square column (five chunks) on a pad footing, under a pier cap
//!   (three chunks across the deck).
//! * Each span is 14.5–15 m of deck in chunks of about 2.5 m x 1.5 m (six along, six
//!   across), with parapets on both edges.
//! * Piers 1 and 3 are seat piers: a 3 m cap with a row of bearing seats under each
//!   span's end, the spans' ends 1 m over the cap, and between them a 1 m link slab, the
//!   top third of the deck's depth.
//! * Pier 2, the one the ball hits, is the hinge pier: a Gerber (half-joint) hinge. Spans
//!   2 and 3 are drop-in spans. They hang there by their bonds alone, on the ends of a
//!   2 m link slab 0.15 m deep (the nibs), with no seat under their ends. The link sits on
//!   a 1 m diaphragm, on one row of bearings.
//! * When the hinge pier's column and the nibs break, those span ends have nothing under
//!   them, while the far ends still rest on their seats. Each span then swings down about
//!   its seat as one slab.
//! * The abutments are supports (fixed, mass 0) under the end spans' bearings, with 1 m
//!   seats.
//! * A lower road crosses under span 2 at grade, where that span's free end comes down:
//!   slabs bonded to a sub-base of supports, for the secondary impact.
//!
//! The ball comes in from the side (+y) at the base of pier 2.
//!
//! Why the hinge pier. In `blast_parity` the struck structure is anchored, so the ball
//! rebounds from it and the hit only breaks bonds; the pier's broken column chunks stay
//! stacked under its cap. A span seated on that pier would stay up, held on contact at
//! both ends, and break at midspan. Drop-in spans without a seat there fall once their
//! bonds go, whatever the column does. The nibs are thinner than the seat piers' links,
//! so they break first, before the far ends release.

use crate::builders::auto_bonds;
use crate::math::Vec3;
use crate::scene::{BondDesc, ChunkDesc};
use crate::wrecking::{between_free, block, footing, joint_of, Aim, Structure, BALLS};

/// The viaduct's dimensions (m). x runs along the bridge, y across it, z up.
pub struct ViaductLayout;

impl ViaductLayout {
    pub const SPANS: usize = 4;
    /// Support line to support line.
    pub const SPAN: f64 = 16.0;
    /// The link slab between the spans' ends over a seat pier, and its depth (the top of
    /// the deck).
    pub const LINK: f64 = 1.0;
    pub const LINK_DEPTH: f64 = 0.3;
    /// The hinge pier's link slab's depth: the nibs the drop-in spans hang on.
    pub const NIB: f64 = 0.15;
    /// The hinge pier's gap between the spans' ends (its link slab's length), and the
    /// diaphragm under the link (the deck's depth below it), narrower than the gap.
    pub const HINGE_GAP: f64 = 2.0;
    pub const DIAPHRAGM: f64 = 1.0;
    /// Deck chunks along a span and across the deck.
    pub const ALONG: usize = 6;
    pub const ACROSS: usize = 6;
    pub const WIDTH: f64 = 9.0;
    pub const DECK: f64 = 0.9;
    pub const DECK_TOP: f64 = 10.0;
    /// Parapets on both edges, one a deck chunk.
    pub const PARAPET_WIDTH: f64 = 0.4;
    pub const PARAPET_HEIGHT: f64 = 1.0;
    /// Bearing seats, one under each deck strip in a row: length along the bridge, width
    /// and height.
    pub const BEARING: [f64; 3] = [0.5, 0.6, 0.3];
    /// Pier caps: length along the bridge (seat piers, the hinge pier), depth, chunks
    /// across.
    pub const SEAT_CAP: f64 = 3.0;
    pub const HINGE_CAP: f64 = 1.8;
    pub const CAP_DEPTH: f64 = 1.8;
    pub const CAP_CHUNKS: usize = 3;
    /// Square column section and its chunks.
    pub const COLUMN: f64 = 1.8;
    pub const COLUMN_CHUNKS: usize = 5;
    /// Pad footing: plan side and depth.
    pub const FOOTING: f64 = 3.6;
    pub const FOOTING_DEPTH: f64 = 0.5;
    /// Abutment seat: behind its support line and in front of it (under the span).
    pub const ABUTMENT_BACK: f64 = 1.0;
    pub const ABUTMENT_SEAT: f64 = 1.5;
    /// The lower road under span 2: from x = `ROAD_X0`, `ROAD_ALONG` slabs of `ROAD_SLAB`
    /// square along x and `ROAD_ACROSS` along y (centred), `ROAD_DEPTH` thick on a
    /// sub-base `ROAD_BASE` thick.
    pub const ROAD_X0: f64 = -11.5;
    pub const ROAD_SLAB: f64 = 3.0;
    pub const ROAD_ALONG: usize = 3;
    pub const ROAD_ACROSS: usize = 10;
    pub const ROAD_HALF_LENGTH: f64 = 0.5 * Self::ROAD_ACROSS as f64 * Self::ROAD_SLAB;
    pub const ROAD_DEPTH: f64 = 0.25;
    pub const ROAD_BASE: f64 = 0.2;
    /// The hinge pier, the one the ball hits (1-based, west to east).
    pub const HINGE_PIER: usize = 2;

    pub const PIERS: usize = Self::SPANS - 1;
    /// Road slabs and their sub-bases.
    pub const ROAD_CHUNKS: usize = 2 * Self::ROAD_ALONG * Self::ROAD_ACROSS;
    /// A span: deck chunks and parapets.
    pub const PER_SPAN: usize = Self::ALONG * Self::ACROSS + 2 * Self::ALONG;
    /// A pier's footing, column chunks and cap chunks.
    pub const PER_PIER: usize = 1 + Self::COLUMN_CHUNKS + Self::CAP_CHUNKS;
    /// A seat pier's two rows of bearings and its link slab; the hinge pier's row of
    /// bearings, its diaphragm and its link slab.
    pub const SEAT: usize = 3 * Self::ACROSS;
    pub const HINGE: usize = 3 * Self::ACROSS;
    /// An abutment: its seat (one support) and a row of bearings.
    pub const PER_ABUTMENT: usize = 1 + Self::ACROSS;
    pub const CHUNKS: usize = Self::SPANS * Self::PER_SPAN + Self::PIERS * Self::PER_PIER + (Self::PIERS - 1) * Self::SEAT + Self::HINGE + 2 * Self::PER_ABUTMENT + Self::ROAD_CHUNKS;

    /// Support line `k` (0 the west abutment, `SPANS` the east one).
    pub fn line(k: usize) -> f64 {
        (k as f64 - 0.5 * Self::SPANS as f64) * Self::SPAN
    }
    /// How far a span's end stops short of support line `k`: half the hinge gap at the
    /// hinge pier, half a link elsewhere.
    pub fn end_gap(k: usize) -> f64 {
        if k == Self::HINGE_PIER {
            0.5 * Self::HINGE_GAP
        } else {
            0.5 * Self::LINK
        }
    }
    /// Span `s`'s deck (0-based), from x0 to x1.
    pub fn span(s: usize) -> (f64, f64) {
        (Self::line(s) + Self::end_gap(s), Self::line(s + 1) - Self::end_gap(s + 1))
    }
    pub fn deck_soffit() -> f64 {
        Self::DECK_TOP - Self::DECK
    }
    pub fn cap_top() -> f64 {
        Self::deck_soffit() - Self::BEARING[2]
    }
    pub fn cap_bottom() -> f64 {
        Self::cap_top() - Self::CAP_DEPTH
    }
    pub fn column_chunk() -> f64 {
        (Self::cap_bottom() - Self::FOOTING_DEPTH) / Self::COLUMN_CHUNKS as f64
    }
}

/// A span's deck as it moves: the deck chunks at its west and east ends and its bonded
/// chunk pairs (deck and parapets), by chunk index.
pub struct SpanProbe {
    /// The span's group, `span-N`.
    pub name: String,
    pub west: Vec<usize>,
    pub east: Vec<usize>,
    pub pairs: Vec<(usize, usize)>,
}

/// Where a span is: how far its west and east ends have dropped (m, the mean of their
/// deck chunks' centres) and how many of its bonded pairs are still at their distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpanState {
    pub west_drop: f64,
    pub east_drop: f64,
    pub intact: usize,
    pub pairs: usize,
}

/// Pairs within this of their distance (m) are intact: moving as one slab.
pub const INTACT: f64 = 0.05;

impl SpanProbe {
    /// The span's state with its chunks' centres at `now` (the stress frame, Z-up), from
    /// `start`.
    pub fn state(&self, start: &[[f64; 3]], now: &[[f64; 3]]) -> SpanState {
        let drop = |ks: &[usize]| ks.iter().map(|&k| start[k][2] - now[k][2]).sum::<f64>() / ks.len().max(1) as f64;
        let distance = |p: &[[f64; 3]], a: usize, b: usize| (Vec3::from_array(p[a]) - Vec3::from_array(p[b])).norm();
        let intact = self.pairs.iter().filter(|&&(a, b)| (distance(now, a, b) - distance(start, a, b)).abs() < INTACT).count();
        SpanState { west_drop: drop(&self.west), east_drop: drop(&self.east), intact, pairs: self.pairs.len() }
    }
}

/// The spans of a structure's chunks and bonds (the viaduct's, or the same chunks
/// imported from its pack): every `span-N` group with deck chunks, in group order.
pub fn span_probes(chunks: &[ChunkDesc], bonds: &[BondDesc]) -> Vec<SpanProbe> {
    let group = |k: usize| chunks[k].groups.get(1).map(String::as_str).unwrap_or("");
    let mut spans: std::collections::BTreeMap<&str, Vec<usize>> = std::collections::BTreeMap::new();
    for (k, c) in chunks.iter().enumerate() {
        if group(k).starts_with("span-") && c.groups.first().is_some_and(|g| g == "deck") {
            spans.entry(group(k)).or_default().push(k);
        }
    }
    spans
        .into_iter()
        .map(|(name, deck)| {
            let x = |k: &usize| chunks[*k].center[0];
            let (lo, hi) = deck.iter().map(x).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
            SpanProbe {
                name: name.to_string(),
                west: deck.iter().copied().filter(|k| x(k) < lo + 1e-9).collect(),
                east: deck.iter().copied().filter(|k| x(k) > hi - 1e-9).collect(),
                pairs: bonds.iter().filter(|b| group(b.a) == name && group(b.b) == name).map(|b| (b.a, b.b)).collect(),
            }
        })
        .collect()
}

/// The viaduct: four 16 m spans of a 9 m deck 10 m up, continuous over three hammerhead
/// piers, the middle one a Gerber hinge for two drop-in spans; two abutments, and a lower
/// road at grade under span 2 (module doc). 347 chunks, 575 bonds. The ball comes in along
/// +y at the hinge pier's column, one column chunk above its footing.
pub fn viaduct() -> Structure {
    type L = ViaductLayout;
    let rc = "reinforced-concrete";
    let hw = 0.5 * L::WIDTH;
    let strip = L::WIDTH / L::ACROSS as f64;
    let y = |k: usize| -hw + k as f64 * strip;
    let (soffit, top, cap_top) = (L::deck_soffit(), L::DECK_TOP, L::cap_top());
    let mut chunks: Vec<ChunkDesc> = Vec::with_capacity(L::CHUNKS);
    // A row of bearing seats from `x0` to `x0 + BEARING[0]`, one under the middle of each
    // deck strip.
    let bearings = |chunks: &mut Vec<ChunkDesc>, x0: f64, group: &str| {
        let (bx, by) = (L::BEARING[0], 0.5 * L::BEARING[1]);
        for k in 0..L::ACROSS {
            let yc = y(k) + 0.5 * strip;
            chunks.push(block([x0, yc - by, cap_top], [x0 + bx, yc + by, soffit], rc, "bearing", group.into()));
        }
    };
    // A row of deck-width chunks over a support line, from x0 to x1 and z0 to the deck's
    // top.
    let strips = |chunks: &mut Vec<ChunkDesc>, x0: f64, x1: f64, z0: f64, kind: &str, group: &str| {
        for k in 0..L::ACROSS {
            chunks.push(block([x0, y(k), z0], [x1, y(k + 1), top], rc, kind, group.into()));
        }
    };
    // Abutments: a seat (a support) at each end, and the end span's bearings on it.
    for (k, group) in [(0, "abutment-W"), (L::SPANS, "abutment-E")] {
        let x = L::line(k);
        let (lo, hi) = if k == 0 { (x - L::ABUTMENT_BACK, x + L::ABUTMENT_SEAT) } else { (x - L::ABUTMENT_SEAT, x + L::ABUTMENT_BACK) };
        chunks.push(footing(block([lo, -hw, 0.0], [hi, hw, cap_top], "footing-anchor", "abutment", group.into())));
        let x0 = if k == 0 { x + L::end_gap(k) } else { x - L::end_gap(k) - L::BEARING[0] };
        bearings(&mut chunks, x0, group);
    }
    // Piers: pad footing, column, cap; on a seat pier a row of bearings under each span's
    // end and the link slab between them, on the hinge pier the diaphragm on its bearings.
    for p in 1..=L::PIERS {
        let (x, group) = (L::line(p), format!("pier-{p}"));
        let hinge = p == L::HINGE_PIER;
        let (hf, hc) = (0.5 * L::FOOTING, 0.5 * L::COLUMN);
        let hcap = 0.5 * if hinge { L::HINGE_CAP } else { L::SEAT_CAP };
        chunks.push(footing(block([x - hf, -hf, 0.0], [x + hf, hf, L::FOOTING_DEPTH], "footing-anchor", "footing", group.clone())));
        for c in 0..L::COLUMN_CHUNKS {
            let z0 = L::FOOTING_DEPTH + c as f64 * L::column_chunk();
            chunks.push(block([x - hc, -hc, z0], [x + hc, hc, z0 + L::column_chunk()], rc, "column", group.clone()));
        }
        // The cap's middle chunk sits on the column; the outer two are its cantilevers.
        let third = L::WIDTH / L::CAP_CHUNKS as f64;
        for c in 0..L::CAP_CHUNKS {
            let y0 = -hw + c as f64 * third;
            chunks.push(block([x - hcap, y0, L::cap_bottom()], [x + hcap, y0 + third, cap_top], rc, "cap", group.clone()));
        }
        if hinge {
            // The hinge: a link slab as long as the gap between the spans, on a diaphragm
            // narrower than the gap, so the spans hang on the link's ends alone.
            bearings(&mut chunks, x - 0.5 * L::BEARING[0], &group);
            let hd = 0.5 * L::DIAPHRAGM;
            for k in 0..L::ACROSS {
                chunks.push(block([x - hd, y(k), soffit], [x + hd, y(k + 1), top - L::NIB], rc, "diaphragm", group.clone()));
            }
            strips(&mut chunks, x - L::end_gap(p), x + L::end_gap(p), top - L::NIB, "link", &group);
        } else {
            bearings(&mut chunks, x - L::end_gap(p) - L::BEARING[0], &group);
            bearings(&mut chunks, x + L::end_gap(p), &group);
            strips(&mut chunks, x - 0.5 * L::LINK, x + 0.5 * L::LINK, top - L::LINK_DEPTH, "link", &group);
        }
    }
    // Spans: the deck between its supports' links (or diaphragm), parapets on its edges.
    for s in 0..L::SPANS {
        let group = format!("span-{}", s + 1);
        let (x0, x1) = L::span(s);
        let piece = (x1 - x0) / L::ALONG as f64;
        for i in 0..L::ALONG {
            let (a, b) = (x0 + i as f64 * piece, x0 + (i + 1) as f64 * piece);
            for k in 0..L::ACROSS {
                chunks.push(block([a, y(k), soffit], [b, y(k + 1), top], rc, "deck", group.clone()));
            }
            for (y0, y1) in [(-hw, -hw + L::PARAPET_WIDTH), (hw - L::PARAPET_WIDTH, hw)] {
                chunks.push(block([a, y0, top], [b, y1, top + L::PARAPET_HEIGHT], rc, "parapet", group.clone()));
            }
        }
    }
    // The lower road under span 2, clear of everything else: slabs on a sub-base (the
    // supports), each slab bonded to its sub-base and its neighbours.
    for j in 0..L::ROAD_ACROSS {
        for i in 0..L::ROAD_ALONG {
            let (x0, y0) = (L::ROAD_X0 + i as f64 * L::ROAD_SLAB, -L::ROAD_HALF_LENGTH + j as f64 * L::ROAD_SLAB);
            let (x1, y1) = (x0 + L::ROAD_SLAB, y0 + L::ROAD_SLAB);
            chunks.push(footing(block([x0, y0, 0.0], [x1, y1, L::ROAD_BASE], "footing-anchor", "footing", "road".into())));
            chunks.push(block([x0, y0, L::ROAD_BASE], [x1, y1, L::ROAD_BASE + L::ROAD_DEPTH], rc, "road", "road".into()));
        }
    }
    debug_assert_eq!(chunks.len(), L::CHUNKS);
    let bonds = between_free(&chunks, auto_bonds(&chunks, joint_of));
    Structure {
        key: "bo_viaduct",
        title: "Highway viaduct",
        chunks,
        bonds,
        aim: Aim { target: [L::line(L::HINGE_PIER), -0.5 * L::COLUMN, L::FOOTING_DEPTH + L::column_chunk()], direction: [0.0, 1.0, 0.0] },
        ball: BALLS[1],
    }
}
