//! Heavy-vehicle parts (trucks, tractors, trailers): each defined once.
//!
//! Every builder returns [`MechPart`]s: a part's nodes with their masses, its
//! internal beams, its mounts (named break groups) and links, and the closed
//! solid drawn through the same points (`generated-vehicles-lessons.md` §1,
//! "one definition, everything derived"). A design made only of such parts
//! (its [`crate::Design::structure`] an empty [`crate::structure::Structure`],
//! no unibody shell) is rigged by the rigger's parts-only path
//! (`engine-softbody-authoring` `gen2::parts_rig`): nodes, beams and masses as
//! defined here, springs from the node budget under the explicit step's
//! bound, and one closed collision hull per part (its nodes' convex hull), so
//! **every part that collides must be convex in its nodes** (a ladder frame is
//! two rails, not one part).
//!
//! glTF frame: metres, +y up, +z forward, +x the vehicle's left.
//!
//! # API (for a truck, a tractor or a trailer)
//!
//! * [`ladder_frame`] (`&LadderFrame`) → `[frame_rail_L, frame_rail_R,
//!   frame_crossmembers]`. Two C-channel rails as closed four-node box sections
//!   at evenly spaced stations (`bays + 1`) between `z_front` and `z_rear`,
//!   chords sized to the steel (yield σ_y·A/4, break σ_u·A/4), cross-members
//!   as links at `cross_at` stations with gusset bracing in the bays beside
//!   them. `equipment_kg` is the frame-mounted mass with no part of its own
//!   (air tanks, wiring, brackets), spread over the rails' nodes. Node names:
//!   `fr{L|R}{station}{ot|it|ib|ob}` (outer/inner, top/bottom).
//! * [`rail_nodes`] (`&LadderFrame`, side, corners, z range) → node names, for
//!   mounting anything on the frame.
//! * [`beam_axle`] (`&BeamAxle`, `&LadderFrame`) → `[axle, wheel_{tag}L,
//!   wheel_{tag}R]`: a rigid beam axle hung on the rails by its spring hangers
//!   (leaf or air springs: the suspension's compliance is the raycast wheel's
//!   spring, `deformable-vehicles.md` §6), and two raycast wheel parts
//!   (`whe{tag}{l|r}_ai` / `_ao` axle nodes, the rigger's raycast wheel;
//!   three rim nodes: a skin's fit takes five). `dual: true` draws twin tyres
//!   and the raycast wheel is as wide as the pair. [`axle_meshes`] gives the
//!   springs', tyres' and brakes' meshes (they ride those parts: one material
//!   a mesh, since a mesh's vertices of another material's primitive are
//!   counted as paint by the lint's coverage rule).
//!   For a trailer: `tag` names the wheel parts (`wheel_{tag}L/R`; the rigger
//!   reads `FL`, `FR`, `RL`, `RR`), `hanger` the spring hangers' reach along
//!   the rails, `seat_x` the springs under the rails.
//! * [`block`]: a closed eight-node box part (an engine, a tank, a box, a
//!   fifth wheel, a kingpin plate) with a complete internal graph.
//! * [`mount`]: a named break group of beams from a part's nodes to nodes of
//!   other parts.
//! * [`cab`] (`&Cab`) → the cab part (24 nodes: three stations of floor,
//!   floor-over-rail, belt and roof nodes) with its shell, glass, doors, grille,
//!   mirrors and steps; the cab's beams yield before the frame's.
//! * [`front_underrun`] (`&Underrun`) → the front bumper with its front
//!   underrun protection beam (UN R93) on brackets to the rails' front ends.
//!
//! **Crash zones** (`crate::crash::Zone`), stated by each builder: the front
//! underrun bumper `FrontLower`, the cab `Cabin`, axles, springs and wheels
//! `Wheel`; the ladder frame `Other`, because a ladder frame has no counterpart
//! among a car body's sections (it is the front load path, the floor and the
//! rear rails at once, and it is not the survival cell); [`block`] takes its
//! part's zone (the truck's engine, tanks, battery box and fifth wheel:
//! `Other`, carried, not structure).
//!
//! Every value is a physical one: masses of the real components, yield and
//! break forces of their sections; only the springs are left to the node
//! budget (the step's bound, lessons §2: heavier nodes allow stiffer beams).

use glam::DVec3;

use crate::crash::Zone;
use crate::mechanical::{vals, BeamKind, Internal, Link, MechPart, Mount, PartNode, Vals};
use crate::mesh::{box_, closed_lathe, rbox, Mat, Mesh};
use crate::parts::{self as p, RimStyle, WheelSpec};

const INF: f64 = f64::INFINITY;
/// Steel density (kg/m³).
pub const RHO_STEEL: f64 = 7850.0;
/// Hardware beams (mounts) are at least this long: the solver keeps a JBeam
/// rest length at 0.1 m or more when a beam yields (lessons §16).
pub const MOUNT_MIN_LEN: f64 = 0.12;

/// A part with its crash zone, stated where the part is defined.
fn part(name: &str, class: &'static str, zone: Zone) -> MechPart {
    MechPart {
        name: name.into(),
        class,
        zone,
        mesh: Mesh::new(),
        nodes: Vec::new(),
        internal: Vec::new(),
        mounts: Vec::new(),
        links: Vec::new(),
        binds: Vec::new(),
        meshes: Vec::new(),
    }
}

fn add_node(p: &mut MechPart, name: String, pos: DVec3, mass: f64, collide: bool) -> usize {
    p.nodes.push(PartNode {
        name,
        pos,
        mass,
        collide,
    });
    p.nodes.len() - 1
}

/// Every pair of the part's nodes.
fn complete(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect()
}

/// A mount: up to `beams` beams (nearest pairs first, each at least
/// [`MOUNT_MIN_LEN`]) from `from` (this part's nodes) to `to`, in break group
/// `group`.
pub fn mount(
    p: &mut MechPart,
    group: &str,
    from: Vec<String>,
    to: Vec<String>,
    beams: usize,
    v: Vals,
) {
    p.mounts.push(Mount {
        group: group.into(),
        from,
        to,
        beams,
        kind: BeamKind::Normal,
        v,
        precomp: 1.0,
    });
}

const CORNERS: [&str; 4] = ["ot", "it", "ib", "ob"];

fn side_tag(s: f64) -> &'static str {
    if s > 0.0 {
        "L"
    } else {
        "R"
    }
}

// -- a rigid box part ---------------------------------------------------------------------------

/// A closed eight-node box part between `lo` and `hi` (its nodes at the
/// corners inset by `inset`), complete internal graph at `v`, its solid a
/// rounded box of `mat`.
pub fn block(
    name: &str,
    class: &'static str,
    lo: DVec3,
    hi: DVec3,
    kg: f64,
    inset: f64,
    mat: Mat,
    v: Vals,
    zone: Zone,
) -> MechPart {
    let mut p = part(name, class, zone);
    let tag = short_tag(name);
    let c = 0.5 * (lo + hi);
    let h = 0.5 * (hi - lo);
    for (k, sx) in [-1.0f64, 1.0].iter().enumerate() {
        for (j, sy) in [-1.0f64, 1.0].iter().enumerate() {
            for (i, sz) in [-1.0f64, 1.0].iter().enumerate() {
                let q = c + DVec3::new(sx * (h.x - inset), sy * (h.y - inset), sz * (h.z - inset));
                add_node(&mut p, format!("{tag}{k}{j}{i}"), q, kg / 8.0, true);
            }
        }
    }
    p.internal.push(Internal {
        pairs: complete(8),
        v,
        deform_group: None,
    });
    p.mesh = rbox(c, h, (0.25 * h.min_element()).min(0.06), 3, mat);
    p
}

/// A part's node-name prefix: its name's first three letters and each later
/// word's initial (`fuel_tank` → `fuet`).
pub fn short_tag(name: &str) -> String {
    let mut it = name.split('_');
    let head: String = it.next().unwrap().chars().take(3).collect();
    let rest: String = it
        .map(|w| {
            if w.len() <= 2 {
                w.to_lowercase()
            } else {
                w.chars().take(1).collect()
            }
        })
        .collect();
    format!("{head}{rest}")
}

mod axle;
mod bumper;
mod cab;
mod frame;
pub use axle::*;
pub use bumper::*;
pub use cab::*;
pub use frame::*;
