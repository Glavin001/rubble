//! Each beam's zone: a section (from the generator's part prefixes, or the body's rest
//! geometry when it names none) and its side.
//!
//! The spawned vehicle keeps the desc's node order (raycast wheels add no nodes; other
//! wheel models add theirs after the desc's) and its beam order: the desc's beams one to
//! one first, then torsion elements, couplers, LBEAMs, the hub rigidity beams of raycast
//! wheels (virtual plain beams), rail segments (`ExplicitBeams::from_desc_jbeam_with`).
//! A beam past the desc's is a wheel beam when it is virtual or touches a node past the
//! desc's, else `other`; its side still comes from its nodes.

use super::super::{Section, Template};
use crate::explicit_beams::{Beam, ExplicitKind, ExplicitType};
use glam::Vec3;

pub const SECTIONS: [(Section, &str); 7] = [
    (Section::FrontLower, "front_lower"),
    (Section::FrontUpper, "front_upper"),
    (Section::Cabin, "cabin"),
    (Section::Door, "door"),
    (Section::Rear, "rear"),
    (Section::Wheel, "wheel"),
    (Section::Other, "other"),
];

pub fn section_index(s: Section) -> usize {
    SECTIONS.iter().position(|(x, _)| *x == s).unwrap_or(SECTIONS.len() - 1)
}

/// A beam's zone: its section and how much of it is on each side (1 and 0, or a half
/// each for a beam on the centre line, so a symmetric body measures symmetric).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zone {
    pub section: Section,
    pub left: f64,
    pub right: f64,
}

/// The section of the longest prefix of `name` in `parts`.
fn by_prefix(parts: &[(String, Section)], name: &str) -> Option<Section> {
    parts.iter().filter(|(p, _)| name.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|(_, s)| *s)
}

/// Every beam's zone. `rest`: each node at rest in the body's own frame (forward +Z, up
/// +Y, left +X; the template's positions, then any added node un-turned from spawn).
pub fn assign(t: &Template, beams: &[Beam], rest: &[Vec3]) -> Vec<Zone> {
    let d = &t.desc;
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in rest.iter().take(d.nodes.len().max(1)) {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let (len, mid) = (hi.z - lo.z, 0.5 * (lo + hi));
    let centre_tol = 1e-3 * (hi.x - lo.x).max(1e-3);
    let named = |i: u32| d.nodes.get(i as usize).map(|n| n.name.as_str());
    beams
        .iter()
        .enumerate()
        .map(|(j, b)| {
            let (pa, pb) = (rest.get(b.a as usize).copied().unwrap_or(mid), rest.get(b.b as usize).copied().unwrap_or(mid));
            let m = 0.5 * (pa + pb);
            let extra_node = named(b.a).is_none() || named(b.b).is_none();
            let section = if extra_node || (j >= d.beams.len() && b.ty == ExplicitType::Virtual && b.kind == ExplicitKind::Plain) {
                Section::Wheel
            } else if j >= d.beams.len() {
                Section::Other
            } else if !t.zones.parts.is_empty() {
                named(b.a).and_then(|n| by_prefix(&t.zones.parts, n)).or_else(|| named(b.b).and_then(|n| by_prefix(&t.zones.parts, n))).unwrap_or(Section::Other)
            } else if m.z > hi.z - 0.30 * len {
                if m.y > mid.y {
                    Section::FrontUpper
                } else {
                    Section::FrontLower
                }
            } else if m.z < lo.z + 0.25 * len {
                Section::Rear
            } else {
                Section::Cabin
            };
            let x = m.x - mid.x;
            let (left, right) = if x.abs() <= centre_tol {
                (0.5, 0.5)
            } else if x > 0.0 {
                (1.0, 0.0)
            } else {
                (0.0, 1.0)
            };
            Zone { section, left, right }
        })
        .collect()
}
