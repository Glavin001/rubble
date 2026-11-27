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
//! Every value is a physical one: masses of the real components, yield and
//! break forces of their sections; only the springs are left to the node
//! budget (the step's bound, lessons §2: heavier nodes allow stiffer beams).

use glam::DVec3;

use crate::mechanical::{vals, BeamKind, Internal, Link, MechPart, Mount, PartNode, Vals};
use crate::mesh::{box_, closed_lathe, rbox, Mat, Mesh};
use crate::parts::{self as p, RimStyle, WheelSpec};

const INF: f64 = f64::INFINITY;
/// Steel density (kg/m³).
pub const RHO_STEEL: f64 = 7850.0;
/// Hardware beams (mounts) are at least this long: the solver keeps a JBeam
/// rest length at 0.1 m or more when a beam yields (lessons §16).
pub const MOUNT_MIN_LEN: f64 = 0.12;

fn part(name: &str, class: &'static str) -> MechPart {
    MechPart {
        name: name.into(),
        class,
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

// -- the ladder frame ---------------------------------------------------------------------------

/// A ladder frame of two C-channel side members.
#[derive(Clone, Debug)]
pub struct LadderFrame {
    pub z_front: f64,
    pub z_rear: f64,
    /// bays between stations (stations = bays + 1, evenly spaced)
    pub bays: usize,
    /// half the frame's outer width (m): the rails' outer webs
    pub half_width: f64,
    /// side member height, flange width and gauge (m)
    pub web_h: f64,
    pub flange_w: f64,
    pub gauge: f64,
    /// the rails' top face (m)
    pub top_y: f64,
    /// steel yield and ultimate strength (Pa)
    pub yield_pa: f64,
    pub ultimate_pa: f64,
    /// stations carrying a cross-member
    pub cross_at: Vec<usize>,
    /// cross-members' mass each (kg)
    pub cross_kg: f64,
    /// frame-mounted mass with no part of its own (kg), spread over the rails
    pub equipment_kg: f64,
}

impl LadderFrame {
    pub fn stations(&self) -> Vec<f64> {
        (0..=self.bays)
            .map(|k| self.z_front + (self.z_rear - self.z_front) * k as f64 / self.bays as f64)
            .collect()
    }
    /// One rail's steel section (m²): web plus two flanges.
    pub fn rail_area(&self) -> f64 {
        (self.web_h + 2.0 * (self.flange_w - self.gauge)) * self.gauge
    }
    pub fn rail_kg(&self) -> f64 {
        self.rail_area() * (self.z_front - self.z_rear) * RHO_STEEL
    }
    pub fn mass(&self) -> f64 {
        2.0 * self.rail_kg() + self.cross_at.len() as f64 * self.cross_kg + self.equipment_kg
    }
    pub fn bottom_y(&self) -> f64 {
        self.top_y - self.web_h
    }
    /// A corner's (x, y) on side `s` (+1 left): `ot`, `it`, `ib`, `ob`.
    pub fn corner(&self, s: f64, c: &str) -> (f64, f64) {
        let xo = s * self.half_width;
        let xi = s * (self.half_width - self.flange_w);
        match c {
            "ot" => (xo, self.top_y),
            "it" => (xi, self.top_y),
            "ib" => (xi, self.bottom_y()),
            "ob" => (xo, self.bottom_y()),
            _ => panic!("no rail corner {c}"),
        }
    }
}

const CORNERS: [&str; 4] = ["ot", "it", "ib", "ob"];

fn side_tag(s: f64) -> &'static str {
    if s > 0.0 {
        "L"
    } else {
        "R"
    }
}

/// A rail node's name.
pub fn rail_node(s: f64, k: usize, c: &str) -> String {
    format!("fr{}{k}{c}", side_tag(s))
}

/// The rail nodes of side `s` at the `corners` of every station in `z_lo..=z_hi`.
pub fn rail_nodes(f: &LadderFrame, s: f64, corners: &[&str], z_lo: f64, z_hi: f64) -> Vec<String> {
    let mut out = Vec::new();
    for (k, z) in f.stations().into_iter().enumerate() {
        if z >= z_lo - 1e-9 && z <= z_hi + 1e-9 {
            for c in corners {
                out.push(rail_node(s, k, c));
            }
        }
    }
    out
}

/// The station nearest `z`.
pub fn station_near(f: &LadderFrame, z: f64) -> usize {
    let st = f.stations();
    (0..st.len())
        .min_by(|&a, &b| (st[a] - z).abs().total_cmp(&(st[b] - z).abs()))
        .unwrap()
}

/// `[frame_rail_L, frame_rail_R, frame_crossmembers]`.
pub fn ladder_frame(f: &LadderFrame) -> Vec<MechPart> {
    let st = f.stations();
    let n_st = st.len();
    let a = f.rail_area();
    // Chords: a quarter of the section each.
    let chord = vals(0.0, 0.0, f.yield_pa * a / 4.0, f.ultimate_pa * a / 4.0);
    // Webs, flanges and the section's own ring: shear, half the chords' force.
    let web = vals(
        0.0,
        0.0,
        0.5 * f.yield_pa * a / 4.0,
        0.5 * f.ultimate_pa * a / 4.0,
    );
    // A flange-width tie (under 0.1 m: it cannot yield in compression, see
    // [`MOUNT_MIN_LEN`]) keeps the section's shape and breaks with the rail.
    let short = vals(0.0, 0.0, INF, f.ultimate_pa * a / 4.0);
    let spacing = (f.z_front - f.z_rear) / f.bays as f64;
    let mut out = Vec::new();
    let n_nodes = 2 * 4 * n_st;
    let equip = f.equipment_kg / n_nodes as f64;
    for s in [1.0f64, -1.0] {
        let mut r = part(&format!("frame_rail_{}", side_tag(s)), "frame");
        for (k, &z) in st.iter().enumerate() {
            let share = if k == 0 || k + 1 == n_st { 0.5 } else { 1.0 };
            let steel = f.rail_area() * spacing * share * RHO_STEEL / 4.0;
            for c in CORNERS {
                let (x, y) = f.corner(s, c);
                add_node(
                    &mut r,
                    rail_node(s, k, c),
                    DVec3::new(x, y, z),
                    steel + equip,
                    true,
                );
            }
        }
        let id = |k: usize, c: usize| 4 * k + c;
        let mut chords = Vec::new();
        let mut webs = Vec::new();
        let mut shorts = Vec::new();
        for k in 0..n_st {
            // the ring: webs (ot-ob, it-ib) and both diagonals are long, the
            // flanges (ot-it, ob-ib) flange-wide
            webs.push((id(k, 0), id(k, 3)));
            webs.push((id(k, 1), id(k, 2)));
            webs.push((id(k, 0), id(k, 2)));
            webs.push((id(k, 1), id(k, 3)));
            shorts.push((id(k, 0), id(k, 1)));
            shorts.push((id(k, 3), id(k, 2)));
            if k + 1 < n_st {
                for c in 0..4 {
                    chords.push((id(k, c), id(k + 1, c)));
                }
                // one diagonal in every face of the bay, zig-zag along the rail
                // (each node's stiffness budget is shared by its beams: fewer
                // webs leave the chords, which carry the bending, stiffer)
                for (c0, c1) in [(0, 3), (1, 2), (0, 1), (3, 2)] {
                    if k % 2 == 0 {
                        webs.push((id(k, c0), id(k + 1, c1)));
                    } else {
                        webs.push((id(k, c1), id(k + 1, c0)));
                    }
                }
            }
        }
        r.internal.push(Internal {
            pairs: chords,
            v: chord,
            deform_group: None,
        });
        r.internal.push(Internal {
            pairs: webs,
            v: web,
            deform_group: None,
        });
        r.internal.push(Internal {
            pairs: shorts,
            v: short,
            deform_group: None,
        });
        r.mesh = rail_mesh(f, s);
        out.push(r);
    }
    // Cross-members: links between the rails' inner corners, and gussets (the
    // plan bracing of the bays beside each, which a gusseted joint holds).
    let mut x = part("frame_crossmembers", "frame");
    let cm = vals(
        0.0,
        0.0,
        0.5 * f.yield_pa * a / 4.0,
        0.5 * f.ultimate_pa * a / 4.0,
    );
    let link = |x: &mut MechPart, a: String, b: String| {
        let name = format!("xm_{a}_{b}");
        x.links.push(Link {
            name,
            a,
            b,
            v: cm,
            group: None,
        });
    };
    let (l, r) = (1.0, -1.0);
    for &k in &f.cross_at {
        for (ca, cb) in [("it", "it"), ("ib", "ib"), ("it", "ib"), ("ib", "it")] {
            link(&mut x, rail_node(l, k, ca), rail_node(r, k, cb));
        }
        for k2 in [k.wrapping_sub(1), k + 1] {
            if k2 >= n_st {
                continue;
            }
            for c in ["it", "ib"] {
                link(&mut x, rail_node(l, k, c), rail_node(r, k2, c));
                link(&mut x, rail_node(r, k, c), rail_node(l, k2, c));
            }
        }
        x.binds.extend(rail_nodes(
            f,
            l,
            &CORNERS,
            st[k].min(st[(k + 1).min(n_st - 1)]),
            st[k.saturating_sub(1)],
        ));
        x.binds.extend(rail_nodes(
            f,
            r,
            &CORNERS,
            st[k].min(st[(k + 1).min(n_st - 1)]),
            st[k.saturating_sub(1)],
        ));
        let (xi, _) = f.corner(1.0, "it");
        let yc = f.top_y - 0.5 * f.web_h;
        x.mesh.add(&box_(
            DVec3::new(0.0, yc, st[k]),
            DVec3::new(xi - 0.002, 0.5 * f.web_h - 0.03, 0.05),
            Mat::TrimBlack,
            None,
        ));
    }
    x.binds.sort();
    x.binds.dedup();
    // The cross-members' steel rides the rails' inner nodes at their stations.
    let mass_each = f.cross_kg / 4.0;
    for k in &f.cross_at {
        for rail in out.iter_mut() {
            for n in rail.nodes.iter_mut() {
                if n.name.ends_with(&format!("{k}it")) || n.name.ends_with(&format!("{k}ib")) {
                    if n.name[3..n.name.len() - 2] == k.to_string() {
                        n.mass += mass_each;
                    }
                }
            }
        }
    }
    out.push(x);
    out
}

/// A side member: web and two flanges (a C open inward), closed solids.
fn rail_mesh(f: &LadderFrame, s: f64) -> Mesh {
    let zc = 0.5 * (f.z_front + f.z_rear);
    let hz = 0.5 * (f.z_front - f.z_rear) + 0.03;
    let xo = s * (f.half_width - 0.5 * f.gauge);
    let mut m = box_(
        DVec3::new(xo, f.top_y - 0.5 * f.web_h, zc),
        DVec3::new(0.5 * f.gauge, 0.5 * f.web_h, hz),
        Mat::TrimBlack,
        None,
    );
    for y in [f.top_y - 0.5 * f.gauge, f.bottom_y() + 0.5 * f.gauge] {
        let xm = s * (f.half_width - 0.5 * f.flange_w);
        m.add(&box_(
            DVec3::new(xm, y, zc),
            DVec3::new(0.5 * f.flange_w - 0.001, 0.5 * f.gauge, hz - 0.001),
            Mat::TrimBlack,
            None,
        ));
    }
    m
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
) -> MechPart {
    let mut p = part(name, class);
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

// -- beam axles and wheels ----------------------------------------------------------------------

/// A rigid beam axle on spring hangers, with its two raycast wheels.
#[derive(Clone, Debug)]
pub struct BeamAxle {
    /// `F` or `R`: the wheel parts are `wheel_{tag}L` / `wheel_{tag}R`
    pub tag: &'static str,
    pub z: f64,
    /// wheel centre height at design (the tyre's unloaded radius)
    pub wheel_y: f64,
    /// track: wheel centre to wheel centre (a dual pair's centre)
    pub track: f64,
    /// the spring seats' half spacing (under the rails)
    pub seat_x: f64,
    /// axle mass with its springs, hubs and brakes (kg)
    pub kg: f64,
    /// the hangers' reach fore and aft of the axle along the rails (m)
    pub hanger: f64,
    pub tyre_d: f64,
    /// one tyre's width
    pub tyre_w: f64,
    pub rim_d: f64,
    pub rim_w: f64,
    /// twin tyres (the raycast wheel is as wide as the pair)
    pub dual: bool,
    /// one wheel position's mass (kg): rim(s), tyre(s), hub
    pub wheel_kg: f64,
    /// the solver's raycast wheel mass (kg), taken out of the wheel part's nodes
    pub raycast_kg: f64,
    /// the axle beam's and the hangers' (yield, break) per beam (N)
    pub axle: (f64, f64),
    pub hangers: (f64, f64),
}

/// The gap between twin tyres (m).
pub const DUAL_GAP: f64 = 0.035;

impl BeamAxle {
    /// The raycast wheel's width: one tyre, or a twin pair and its gap.
    pub fn contact_w(&self) -> f64 {
        if self.dual {
            2.0 * self.tyre_w + DUAL_GAP
        } else {
            self.tyre_w
        }
    }
}

/// `[axle_{tag}, wheel_{tag}L, wheel_{tag}R]`, hung on `frame`'s rails.
pub fn beam_axle(a: &BeamAxle, f: &LadderFrame) -> Vec<MechPart> {
    let name = if a.tag == "F" {
        "axle_front"
    } else {
        "axle_rear"
    };
    let mut ax = part(name, "suspension");
    let t = a.tag.to_lowercase();
    let y = a.wheel_y;
    for s in [1.0f64, -1.0] {
        let sd = side_tag(s).to_lowercase();
        let x = s * a.seat_x;
        add_node(
            &mut ax,
            format!("ax{t}{sd}_sf"),
            DVec3::new(x, y + 0.13, a.z + 0.11),
            a.kg / 8.0,
            false,
        );
        add_node(
            &mut ax,
            format!("ax{t}{sd}_sr"),
            DVec3::new(x, y + 0.13, a.z - 0.11),
            a.kg / 8.0,
            false,
        );
        add_node(
            &mut ax,
            format!("ax{t}{sd}_sb"),
            DVec3::new(x, y - 0.07, a.z),
            a.kg / 8.0,
            false,
        );
        add_node(
            &mut ax,
            format!("ax{t}{sd}_kp"),
            DVec3::new(s * (a.seat_x + 0.26), y, a.z),
            a.kg / 8.0,
            false,
        );
    }
    ax.internal.push(Internal {
        pairs: complete(8),
        v: vals(0.0, 0.0, a.axle.0, a.axle.1),
        deform_group: None,
    });
    for s in [1.0f64, -1.0] {
        let sd = side_tag(s).to_lowercase();
        let from = ["sf", "sr", "sb"]
            .iter()
            .map(|c| format!("ax{t}{sd}_{c}"))
            .collect();
        let to = rail_nodes(f, s, &["ib", "ob", "it"], a.z - a.hanger, a.z + a.hanger);
        mount(
            &mut ax,
            &format!("spring_{}{}", a.tag, side_tag(s)),
            from,
            to,
            12,
            vals(0.0, 0.0, a.hangers.0, a.hangers.1),
        );
    }
    ax.mesh = axle_mesh(a);
    ax.meshes = vec![format!("springs_{}", a.tag)];
    let mut out = vec![ax];
    for s in [1.0f64, -1.0] {
        let corner = format!("{}{}", a.tag, side_tag(s));
        let mut w = part(&format!("wheel_{corner}"), "wheel");
        let tag = format!("whe{}", corner.to_lowercase());
        let c = s * 0.5 * a.track;
        let hw = 0.5 * a.contact_w() - 0.03;
        // two axle nodes and three on the rim (a skin's fit takes five)
        let own = (a.wheel_kg - a.raycast_kg) / 5.0;
        add_node(
            &mut w,
            format!("{tag}_ai"),
            DVec3::new(c - s * hw, y, a.z),
            own,
            false,
        );
        add_node(
            &mut w,
            format!("{tag}_ao"),
            DVec3::new(c + s * hw, y, a.z),
            own,
            false,
        );
        add_node(
            &mut w,
            format!("{tag}_t"),
            DVec3::new(c - s * hw, y + 0.25, a.z),
            own,
            false,
        );
        add_node(
            &mut w,
            format!("{tag}_f"),
            DVec3::new(c - s * hw, y - 0.125, a.z + 0.2165),
            own,
            false,
        );
        add_node(
            &mut w,
            format!("{tag}_r"),
            DVec3::new(c - s * hw, y - 0.125, a.z - 0.2165),
            own,
            false,
        );
        w.internal.push(Internal {
            pairs: complete(5),
            v: vals(0.0, 0.0, INF, a.axle.1),
            deform_group: None,
        });
        let sd = side_tag(s).to_lowercase();
        let to = ["kp", "sf", "sr", "sb"]
            .iter()
            .map(|c| format!("ax{t}{sd}_{c}"))
            .collect();
        let from = w.nodes.iter().map(|n| n.name.clone()).collect();
        mount(
            &mut w,
            &format!("hub_{corner}"),
            from,
            to,
            16,
            vals(0.0, 0.0, a.axle.0, a.axle.1),
        );
        // the rim(s); the tyre(s) and the brake ride this part too ([`axle_meshes`])
        let offs: Vec<f64> = if a.dual {
            vec![-0.5 * (a.tyre_w + DUAL_GAP), 0.5 * (a.tyre_w + DUAL_GAP)]
        } else {
            vec![0.0]
        };
        let mut rim = Mesh::new();
        for (i, o) in offs.iter().enumerate() {
            let mut ri = if s > 0.0 {
                p::rim_steel(a.rim_d, a.rim_w, 64)
            } else {
                let mut m = p::rim_steel(a.rim_d, a.rim_w, 64);
                m.mirror_x();
                m
            };
            // the inner of a twin pair is turned round: back to back with the outer
            if a.dual && i == 0 {
                ri.mirror_x();
            }
            ri.translate(DVec3::new(c + s * o, y, a.z));
            rim.add(&ri);
        }
        w.mesh = rim;
        w.meshes = vec![format!("tyre_{corner}"), format!("brake_disc_{corner}")];
        out.push(w);
    }
    out
}

/// The other meshes of [`beam_axle`]'s parts, by name: the springs (riding
/// the axle) and each wheel's tyre(s) and brake (riding the wheel parts:
/// `MechPart::meshes`). One material a mesh.
pub fn axle_meshes(a: &BeamAxle, f: &LadderFrame) -> Vec<(String, Mesh)> {
    let spec = WheelSpec {
        track: a.track,
        wheel_y: a.wheel_y,
        tyre_d: a.tyre_d,
        tyre_w: a.tyre_w,
        rim_d: a.rim_d,
        rim_w: a.rim_w,
        rim_style: RimStyle::Steel,
        disc_d: 0.43,
    };
    let mut out = Vec::new();
    // the springs: a leaf pack (or an air spring's beam) under each rail
    let mut springs = Mesh::new();
    for s in [1.0f64, -1.0] {
        let x = s * a.seat_x;
        let ytop = f.bottom_y() - 0.02;
        let ybot = a.wheel_y + 0.08;
        let hz = (a.hanger * 0.8).min(0.75);
        springs.add(&box_(
            DVec3::new(x, 0.5 * (ytop + ybot), a.z),
            DVec3::new(0.045, 0.5 * (ytop - ybot).max(0.03), hz),
            Mat::MetalBare,
            None,
        ));
    }
    out.push((format!("springs_{}", a.tag), springs));
    for s in [1.0f64, -1.0] {
        let corner = format!("{}{}", a.tag, side_tag(s));
        let offs: Vec<f64> = if a.dual {
            vec![-0.5 * (a.tyre_w + DUAL_GAP), 0.5 * (a.tyre_w + DUAL_GAP)]
        } else {
            vec![0.0]
        };
        let (mut tyre, mut disc) = (Mesh::new(), Mesh::new());
        for (i, o) in offs.iter().enumerate() {
            let (_, mut ty, di, _) = p::wheel(&spec, s, a.z);
            ty.translate(DVec3::new(s * o, 0.0, 0.0));
            tyre.add(&ty);
            if i == 0 {
                disc.add(&di);
            }
        }
        out.push((format!("tyre_{corner}"), tyre));
        out.push((format!("brake_disc_{corner}"), disc));
    }
    out
}

fn axle_mesh(a: &BeamAxle) -> Mesh {
    let y = a.wheel_y;
    let kp = a.seat_x + 0.26;
    let mut m = Mesh::new();
    if a.tag == "F" {
        // a forged I-beam dropped between the spring seats, kingpin bosses
        m.add(&box_(
            DVec3::new(0.0, y - 0.06, a.z),
            DVec3::new(kp - 0.04, 0.05, 0.045),
            Mat::TrimBlack,
            None,
        ));
        for s in [1.0f64, -1.0] {
            m.add(&box_(
                DVec3::new(s * kp, y, a.z),
                DVec3::new(0.05, 0.11, 0.06),
                Mat::TrimBlack,
                None,
            ));
        }
    } else {
        // a banjo housing: tubes to the hubs and the differential's bowl
        let mut tube = closed_lathe(
            &[(0.0001, -kp), (0.075, -kp), (0.075, kp), (0.0001, kp)],
            24,
            Mat::TrimBlack,
        );
        tube.translate(DVec3::new(0.0, y, a.z));
        m.add(&tube);
        m.add(&rbox(
            DVec3::new(0.0, y, a.z + 0.05),
            DVec3::new(0.24, 0.24, 0.20),
            0.08,
            3,
            Mat::TrimBlack,
        ));
    }
    m
}

// -- the cab --------------------------------------------------------------------------------------

/// A cab: a box of three stations (front, door pillar, back) on four mounts.
#[derive(Clone, Debug)]
pub struct Cab {
    pub z_front: f64,
    pub z_rear: f64,
    /// the door's rear edge (the B-pillar station)
    pub z_pillar: f64,
    pub floor_y: f64,
    pub belt_y: f64,
    pub roof_y: f64,
    pub half_w: f64,
    /// the shell's edge radius (m)
    pub radius: f64,
    /// complete cab: shell, glass, seats, interior, driver (kg)
    pub kg: f64,
    /// the inner floor nodes' x (over the rails)
    pub mount_x: f64,
    /// the cab's members (yield, break) per beam (N): pillars and roof, floor
    pub members: (f64, f64),
    pub floor: (f64, f64),
}

/// A node on a rounded box's surface toward a corner direction `s` (each
/// component −1, 0 or 1).
fn on_round(c: DVec3, h: DVec3, r: f64, s: DVec3) -> DVec3 {
    let inner = h - DVec3::splat(r);
    let q = (s * h).clamp(-inner, inner);
    let d = s.normalize_or_zero();
    c + q + d * (r - 0.004)
}

/// The cab part and its other meshes (glass, doors, grille, mirrors, steps,
/// visor), which ride it.
pub fn cab(c: &Cab, paint: Mat) -> (MechPart, Vec<(String, Mesh)>) {
    let mut p = part("cab", "cab");
    let ctr = DVec3::new(
        0.0,
        0.5 * (c.floor_y + c.roof_y),
        0.5 * (c.z_front + c.z_rear),
    );
    let h = DVec3::new(
        c.half_w,
        0.5 * (c.roof_y - c.floor_y),
        0.5 * (c.z_front - c.z_rear),
    );
    let n_nodes = 24.0;
    let zs = [c.z_front, c.z_pillar, c.z_rear];
    for (k, &z) in zs.iter().enumerate() {
        // the station's z direction for the corner placement (the pillar's
        // station is on the side faces, not at a corner)
        let sz = if k == 0 {
            1.0
        } else if k == 2 {
            -1.0
        } else {
            0.0
        };
        for s in [1.0f64, -1.0] {
            let sd = side_tag(s);
            let at = |sy: f64, y: f64| {
                let mut q = on_round(ctr, h, c.radius, DVec3::new(s, sy, sz));
                if sy == 0.0 {
                    q.y = y;
                }
                if sz == 0.0 {
                    q.z = z;
                }
                q
            };
            add_node(
                &mut p,
                format!("cab{k}f{sd}"),
                at(-1.0, 0.0),
                c.kg / n_nodes,
                true,
            );
            let mut fi = DVec3::new(s * c.mount_x, c.floor_y + 0.004, z);
            if sz != 0.0 {
                fi.z = on_round(ctr, h, c.radius, DVec3::new(0.0, -1.0, sz)).z;
            }
            add_node(&mut p, format!("cab{k}i{sd}"), fi, c.kg / n_nodes, true);
            add_node(
                &mut p,
                format!("cab{k}b{sd}"),
                at(0.0, c.belt_y),
                c.kg / n_nodes,
                true,
            );
            add_node(
                &mut p,
                format!("cab{k}r{sd}"),
                at(1.0, 0.0),
                c.kg / n_nodes,
                true,
            );
        }
    }
    // Members: each station's ring complete, and every node to every node of
    // the next station. The floor's own beams (both ends on the floor) carry
    // the floor's values.
    let n = p.nodes.len();
    let floor_node = |i: usize| {
        let nm = &p.nodes[i].name;
        nm.as_bytes()[4] == b'f' || nm.as_bytes()[4] == b'i'
    };
    let station = |i: usize| i / 8;
    let (mut members, mut floor) = (Vec::new(), Vec::new());
    for (a, b) in complete(n) {
        if station(b) > station(a) + 1 {
            continue;
        }
        if floor_node(a) && floor_node(b) {
            floor.push((a, b));
        } else {
            members.push((a, b));
        }
    }
    p.internal.push(Internal {
        pairs: members,
        v: vals(0.0, 0.0, c.members.0, c.members.1),
        deform_group: None,
    });
    p.internal.push(Internal {
        pairs: floor,
        v: vals(0.0, 0.0, c.floor.0, c.floor.1),
        deform_group: None,
    });
    p.mesh = rbox(ctr, h, c.radius, 6, paint);
    // -- what rides the cab
    let mut extra: Vec<(String, Mesh)> = Vec::new();
    let zf = c.z_front;
    let wi = c.half_w - c.radius - 0.02;
    // the windscreen, its glass proud of the front face
    let (ws0, ws1) = (c.belt_y + 0.06, c.roof_y - 0.20);
    // (each pane over a black backing on the shell: the cab's inside, seen
    // through the glass, is dark, not the paint)
    let mut backing = Mesh::new();
    extra.push((
        "glass_windshield".into(),
        box_(
            DVec3::new(0.0, 0.5 * (ws0 + ws1), zf + 0.009),
            DVec3::new(wi, 0.5 * (ws1 - ws0), 0.005),
            Mat::Glass,
            None,
        ),
    ));
    backing.add(&box_(
        DVec3::new(0.0, 0.5 * (ws0 + ws1), zf + 0.0015),
        DVec3::new(wi, 0.5 * (ws1 - ws0), 0.0012),
        Mat::TrimBlack,
        None,
    ));
    // the sun visor over it
    extra.push((
        "visor".into(),
        box_(
            DVec3::new(0.0, c.roof_y - 0.12, zf + 0.06),
            DVec3::new(wi + 0.02, 0.025, 0.07),
            Mat::TrimBlack,
            None,
        ),
    ));
    // the grille under it, and the badge bar
    let (g0, g1) = (c.floor_y + 0.06, c.belt_y - 0.06);
    let mut grille = box_(
        DVec3::new(0.0, 0.5 * (g0 + g1), zf + 0.006),
        DVec3::new(wi - 0.10, 0.5 * (g1 - g0), 0.005),
        Mat::TrimBlack,
        None,
    );
    for k in 1..6 {
        let y = g0 + (g1 - g0) * k as f64 / 6.0;
        grille.add(&box_(
            DVec3::new(0.0, y, zf + 0.013),
            DVec3::new(wi - 0.12, 0.012, 0.004),
            Mat::Chrome,
            None,
        ));
    }
    extra.push(("grille".into(), grille));
    for s in [1.0f64, -1.0] {
        let sd = side_tag(s);
        let xs = s * (c.half_w + 0.006);
        // the door: the panel under the belt and the window over it
        let (d0, d1) = (c.z_pillar + 0.05, c.z_front - c.radius - 0.08);
        extra.push((
            format!("door_{sd}"),
            box_(
                DVec3::new(xs, 0.5 * (c.floor_y + 0.08 + c.belt_y), 0.5 * (d0 + d1)),
                DVec3::new(0.005, 0.5 * (c.belt_y - c.floor_y - 0.08), 0.5 * (d1 - d0)),
                paint,
                None,
            ),
        ));
        let (w0, w1) = (c.belt_y + 0.05, c.roof_y - 0.26);
        extra.push((
            format!("glass_door_{sd}"),
            box_(
                DVec3::new(xs + s * 0.003, 0.5 * (w0 + w1), 0.5 * (d0 + d1) + 0.03),
                DVec3::new(0.005, 0.5 * (w1 - w0), 0.5 * (d1 - d0) - 0.06),
                Mat::Glass,
                None,
            ),
        ));
        backing.add(&box_(
            DVec3::new(
                s * (c.half_w + 0.0015),
                0.5 * (w0 + w1),
                0.5 * (d0 + d1) + 0.03,
            ),
            DVec3::new(0.0012, 0.5 * (w1 - w0), 0.5 * (d1 - d0) - 0.06),
            Mat::TrimBlack,
            None,
        ));
        // the door's handle
        extra.push((
            format!("handle_{sd}"),
            box_(
                DVec3::new(xs + s * 0.01, c.belt_y - 0.15, d0 + 0.12),
                DVec3::new(0.008, 0.02, 0.08),
                Mat::TrimBlack,
                None,
            ),
        ));
        // the mirror: an arm from the front pillar and a tall head
        let mut mir = box_(
            DVec3::new(s * (c.half_w + 0.12), c.belt_y + 0.25, zf - 0.08),
            DVec3::new(0.12, 0.015, 0.015),
            Mat::TrimBlack,
            None,
        );
        mir.add(&rbox(
            DVec3::new(s * (c.half_w + 0.25), c.belt_y + 0.20, zf - 0.10),
            DVec3::new(0.04, 0.22, 0.08),
            0.02,
            2,
            Mat::TrimBlack,
        ));
        extra.push((format!("mirror_{sd}"), mir));
        // the steps behind the front wheel, under the door's rear edge (two
        // treads in a black frame)
        let zst = c.z_pillar - 0.22;
        let mut st = Mesh::new();
        for (k, y) in [c.floor_y - 0.25, c.floor_y - 0.55].iter().enumerate() {
            let inset = 0.08 + 0.05 * k as f64;
            st.add(&box_(
                DVec3::new(s * (c.half_w - inset - 0.10), *y, zst),
                DVec3::new(0.10, 0.02, 0.11),
                Mat::TrimBlack,
                None,
            ));
        }
        for dz in [0.13, -0.13] {
            st.add(&box_(
                DVec3::new(s * (c.half_w - 0.05), c.floor_y - 0.30, zst + dz),
                DVec3::new(0.02, 0.30, 0.02),
                Mat::TrimBlack,
                None,
            ));
        }
        extra.push((format!("steps_{sd}"), st));
    }
    // the back wall's small window
    let (r0, r1) = (c.belt_y + 0.10, c.roof_y - 0.35);
    extra.push((
        "glass_rear".into(),
        box_(
            DVec3::new(0.0, 0.5 * (r0 + r1), c.z_rear - 0.009),
            DVec3::new(0.35, 0.5 * (r1 - r0), 0.005),
            Mat::Glass,
            None,
        ),
    ));
    backing.add(&box_(
        DVec3::new(0.0, 0.5 * (r0 + r1), c.z_rear - 0.0015),
        DVec3::new(0.35, 0.5 * (r1 - r0), 0.0012),
        Mat::TrimBlack,
        None,
    ));
    extra.push(("window_backing".into(), backing));
    // roof marker lamps
    let mut roof_lamps = Mesh::new();
    for x in [-0.5, -0.25, 0.0, 0.25, 0.5] {
        roof_lamps.add(&box_(
            DVec3::new(x, c.roof_y + 0.012, zf - 0.20),
            DVec3::new(0.04, 0.012, 0.02),
            Mat::LampEmissive,
            None,
        ));
    }
    extra.push(("roof_lamps".into(), roof_lamps));
    p.meshes = extra.iter().map(|e| e.0.clone()).collect();
    (p, extra)
}

// -- the front bumper and its underrun beam -------------------------------------------------------

/// The front bumper: a steel front underrun protection beam (UN R93: its
/// lower edge at most 400 mm over the ground) under a plastic cover, on two
/// brackets to the rails' front ends.
#[derive(Clone, Debug)]
pub struct Underrun {
    pub z_front: f64,
    pub depth: f64,
    pub lower_y: f64,
    pub upper_y: f64,
    pub half_w: f64,
    pub kg: f64,
    /// the beam's own beams (yield, break) per beam (N)
    pub beam: (f64, f64),
    /// each bracket's beams (yield, break) per beam (N); `bracket_beams` a side
    pub bracket: (f64, f64),
    pub bracket_beams: usize,
}

pub fn front_underrun(
    u: &Underrun,
    f: &LadderFrame,
    paint: Mat,
) -> (MechPart, Vec<(String, Mesh)>) {
    let mut p = part("bumper_front", "bumper");
    let zb = u.z_front - u.depth;
    let xs = [-u.half_w, 0.0, u.half_w];
    for (i, x) in xs.iter().enumerate() {
        for (j, y) in [u.lower_y, u.upper_y].iter().enumerate() {
            for (k, z) in [zb, u.z_front].iter().enumerate() {
                let xi = x - x.signum() * 0.01;
                add_node(
                    &mut p,
                    format!("bf{i}{j}{k}"),
                    DVec3::new(
                        xi,
                        y + if j == 0 { 0.005 } else { -0.005 },
                        z + if k == 0 { 0.005 } else { -0.005 },
                    ),
                    u.kg / 12.0,
                    true,
                );
            }
        }
    }
    // each column's box complete, and each column to the next
    let mut pairs = Vec::new();
    for (a, b) in complete(12) {
        if (b / 4) <= (a / 4) + 1 {
            pairs.push((a, b));
        }
    }
    p.internal.push(Internal {
        pairs,
        v: vals(0.0, 0.0, u.beam.0, u.beam.1),
        deform_group: None,
    });
    for s in [1.0f64, -1.0] {
        let col = if s > 0.0 { 2 } else { 0 };
        let from: Vec<String> = (0..4)
            .map(|q| format!("bf{col}{}{}", q / 2, q % 2))
            .chain((0..4).map(|q| format!("bf1{}{}", q / 2, q % 2)))
            .collect();
        let to = rail_nodes(f, s, &CORNERS, f.z_front - 0.7, f.z_front);
        mount(
            &mut p,
            &format!("fup_bracket_{}", side_tag(s)),
            from,
            to,
            u.bracket_beams,
            vals(0.0, 0.0, u.bracket.0, u.bracket.1),
        );
    }
    // the cover over the beam; the beam's face shows under it
    let c = DVec3::new(
        0.0,
        0.5 * (u.lower_y + u.upper_y),
        u.z_front - 0.5 * u.depth,
    );
    let hh = DVec3::new(u.half_w, 0.5 * (u.upper_y - u.lower_y), 0.5 * u.depth);
    p.mesh = rbox(
        c + DVec3::new(0.0, 0.045, 0.0),
        hh - DVec3::new(0.0, 0.045, 0.0),
        0.05,
        4,
        paint,
    );
    let mut extra = Vec::new();
    extra.push((
        "underrun_beam".into(),
        box_(
            DVec3::new(0.0, u.lower_y + 0.06, u.z_front - 0.5 * u.depth),
            DVec3::new(u.half_w - 0.04, 0.06, 0.5 * u.depth - 0.01),
            Mat::MetalBare,
            None,
        ),
    ));
    for s in [1.0f64, -1.0] {
        let sd = side_tag(s);
        let x = s * (u.half_w - 0.30);
        let y = u.upper_y - 0.16;
        let mut housing = box_(
            DVec3::new(x, y, u.z_front + 0.004),
            DVec3::new(0.22, 0.07, 0.004),
            Mat::TrimBlack,
            None,
        );
        housing.add(&box_(
            DVec3::new(s * (u.half_w - 0.62), u.lower_y + 0.20, u.z_front + 0.004),
            DVec3::new(0.06, 0.04, 0.004),
            Mat::TrimBlack,
            None,
        ));
        extra.push((format!("headlight_housing_{sd}"), housing));
        let mut lens = box_(
            DVec3::new(x, y, u.z_front + 0.011),
            DVec3::new(0.20, 0.06, 0.003),
            Mat::LampLens,
            None,
        );
        lens.add(&box_(
            DVec3::new(s * (u.half_w - 0.62), u.lower_y + 0.20, u.z_front + 0.011),
            DVec3::new(0.05, 0.03, 0.003),
            Mat::LampLens,
            None,
        ));
        extra.push((format!("headlight_{sd}"), lens));
    }
    p.meshes = extra.iter().map(|e: &(String, Mesh)| e.0.clone()).collect();
    (p, extra)
}
