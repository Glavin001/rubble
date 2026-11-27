//! A semi-trailer's own parts: the cargo box on the frame, its rear door frame, and the
//! landing gear it stands on when it is not coupled.

use super::*;

// -- the cargo box ------------------------------------------------------------------------------

/// A van body: a light, stiff shell of floor, walls and roof (its nodes on the floor's
/// and the roof's long edges at evenly spaced stations, and on the floor over each rail),
/// carrying the floor and any load on its floor nodes, on the frame's rails.
#[derive(Clone, Debug)]
pub struct CargoBox {
    pub z_front: f64,
    pub z_rear: f64,
    /// bays between stations (stations = bays + 1)
    pub bays: usize,
    /// the floor structure's underside (the crossmembers' bottom) and the roof (m)
    pub floor_y: f64,
    pub roof_y: f64,
    pub half_w: f64,
    /// the floor nodes over the rails (±x, m): the crossmembers' seats on them
    pub rail_x: f64,
    /// floor and crossmembers, and any load on the deck (kg, on the floor nodes);
    /// walls, roof, front wall (kg, on the roof nodes and the floor nodes alike)
    pub floor_kg: f64,
    pub load_kg: f64,
    pub shell_kg: f64,
    /// the shell's members (yield, break) per beam (N)
    pub members: (f64, f64),
    /// the floor's ties to the rails (yield, break) per beam (N)
    pub ties: (f64, f64),
}

impl CargoBox {
    pub fn stations(&self) -> Vec<f64> {
        (0..=self.bays)
            .map(|k| self.z_front + (self.z_rear - self.z_front) * k as f64 / self.bays as f64)
            .collect()
    }
}

/// A box node's name: station `k`, `f` floor's edge / `i` floor over the rail / `r` roof,
/// side.
pub fn box_node(k: usize, c: char, s: f64) -> String {
    format!("bx{k:02}{c}{}", side_tag(s))
}

/// Nodes a box station has.
const BOX_RING: usize = 6;

/// The cargo box (class `cargo_box`, zone `Other`: carried, not a car's section), tied
/// to `f`'s rails: each floor edge node to the rail's outer top and bottom corners (a
/// triangle, so the floor is held up, not only along), and each floor node over a rail to
/// the rail's corners under it and beside it (the crossmembers' seats: the deck's load
/// goes into the rails there; with the edge ties alone the rails over the tandem stood
/// 18 mm off the box's line parked).
pub fn cargo_box(b: &CargoBox, f: &LadderFrame, paint: Mat) -> MechPart {
    let mut p = part("cargo_box", "cargo_box", Zone::Other);
    let st = b.stations();
    let n_st = st.len() as f64;
    // the floor and the load on the four floor nodes of a station alike; the shell on the
    // edge and roof nodes
    let deck = (b.floor_kg + b.load_kg) / (4.0 * n_st);
    let shell = b.shell_kg / (4.0 * n_st);
    for (k, &z) in st.iter().enumerate() {
        for s in [1.0f64, -1.0] {
            add_node(&mut p, box_node(k, 'f', s), DVec3::new(s * b.half_w, b.floor_y, z), deck + shell, true);
            // (inside the floor's face, under the box's skin: not a collision node, so the
            // skin, the box's convex hull, stays over the edge nodes alone)
            add_node(&mut p, box_node(k, 'i', s), DVec3::new(s * b.rail_x, b.floor_y, z), deck, false);
            add_node(&mut p, box_node(k, 'r', s), DVec3::new(s * b.half_w, b.roof_y, z), shell, true);
        }
    }
    // each station's ring complete, and every node to every node of the next station
    let mut pairs = Vec::new();
    for (a, c) in complete(p.nodes.len()) {
        if c / BOX_RING <= a / BOX_RING + 1 {
            pairs.push((a, c));
        }
    }
    p.internal.push(Internal {
        pairs,
        v: vals(0.0, 0.0, b.members.0, b.members.1),
        deform_group: None,
    });
    for s in [1.0f64, -1.0] {
        let from = (0..st.len()).map(|k| box_node(k, 'f', s)).collect();
        let to = rail_nodes(f, s, &["ot", "ob"], f.z_rear, f.z_front);
        mount(&mut p, &format!("box_floor_{}", side_tag(s)), from, to, 2 * st.len(), vals(0.0, 0.0, b.ties.0, b.ties.1));
        let from = (0..st.len()).map(|k| box_node(k, 'i', s)).collect();
        let to = rail_nodes(f, s, &CORNERS, f.z_rear, f.z_front);
        mount(&mut p, &format!("box_seats_{}", side_tag(s)), from, to, 3 * st.len(), vals(0.0, 0.0, b.ties.0, b.ties.1));
    }
    let c = DVec3::new(0.0, 0.5 * (b.floor_y + b.roof_y), 0.5 * (b.z_front + b.z_rear));
    let h = DVec3::new(b.half_w, 0.5 * (b.roof_y - b.floor_y), 0.5 * (b.z_front - b.z_rear));
    p.mesh = rbox(c, h - DVec3::splat(0.004), 0.03, 3, paint);
    p
}

/// The rear door frame: posts, header and the rear sill at deck height (real structure:
/// what a car meets above the guard), as a closed eight-node box behind the cargo box,
/// on the box's last station and the rails' rear ends.
#[derive(Clone, Debug)]
pub struct RearFrame {
    /// the trailer's rear extremity and the frame's depth along z (m)
    pub z_rear: f64,
    pub depth: f64,
    /// the sill's underside and the header's top (m)
    pub sill_y: f64,
    pub top_y: f64,
    pub half_w: f64,
    /// frame, sill and both doors (kg)
    pub kg: f64,
    /// its members (yield, break) per beam (N), and its mounts'
    pub members: (f64, f64),
    pub mounts: (f64, f64),
}

/// The rear frame (class `rear_frame`, zone `Other`).
pub fn rear_frame(r: &RearFrame, b: &CargoBox, f: &LadderFrame) -> MechPart {
    let lo = DVec3::new(-r.half_w, r.sill_y, r.z_rear);
    let hi = DVec3::new(r.half_w, r.top_y, r.z_rear + r.depth);
    let mut p = block(
        "rear_frame",
        "rear_frame",
        lo,
        hi,
        r.kg,
        0.005,
        Mat::MetalBare,
        vals(0.0, 0.0, r.members.0, r.members.1),
        Zone::Other,
    );
    let tag = short_tag("rear_frame");
    let own: Vec<String> = p.nodes.iter().map(|n| n.name.clone()).collect();
    debug_assert!(own.iter().all(|n| n.starts_with(&tag)));
    let last = b.stations().len() - 1;
    let to: Vec<String> = [1.0f64, -1.0]
        .iter()
        .flat_map(|&s| [box_node(last, 'f', s), box_node(last, 'r', s), box_node(last - 1, 'f', s), box_node(last - 1, 'r', s)])
        .collect();
    mount(&mut p, "rear_frame_box", own.clone(), to, 16, vals(0.0, 0.0, r.mounts.0, r.mounts.1));
    for s in [1.0f64, -1.0] {
        let rails = rail_nodes(f, s, &CORNERS, f.z_rear - 0.01, f.z_rear + 0.01);
        let from = own.iter().filter(|n| (n.as_bytes()[tag.len()] == b'1') == (s > 0.0)).cloned().collect();
        mount(&mut p, &format!("rear_frame_rail_{}", side_tag(s)), from, rails, 4, vals(0.0, 0.0, r.mounts.0, r.mounts.1));
    }
    p
}

// -- the landing gear ---------------------------------------------------------------------------

/// Two landing-gear legs (telescopic legs on sand shoes) under the rails, cross-braced:
/// what a parked semi-trailer stands on in front.
#[derive(Clone, Debug)]
pub struct LandingGear {
    /// the legs' station along the rails, and their x (±, m)
    pub z: f64,
    pub x: f64,
    /// a leg's section (x, z) and its sand shoe's (m)
    pub leg: f64,
    pub kg: f64,
    /// each leg's beams (yield, break) per beam (N), its brackets' to the rail, and the
    /// cross brace's
    pub legs: (f64, f64),
    pub brackets: (f64, f64),
    pub brace: (f64, f64),
    /// Other nodes the brackets reach besides the rails' (the floor structure over the
    /// legs: on a van the landing gear bolts to its crossmembers), per side (+x, −x).
    pub mount_to: [Vec<String>; 2],
}

/// `[landing_gear_L, landing_gear_R]` (class `landing_gear`, zone `Other`): each leg a
/// closed box from the ground (its sand shoe's four nodes on it) to the rail's bottom, on
/// brackets to the rail's nodes at the nearest station and the ones beside it, the two
/// cross-braced; and the cross shaft's mesh (riding the left leg).
pub fn landing_gear(g: &LandingGear, f: &LadderFrame) -> (Vec<MechPart>, Vec<(String, Mesh)>) {
    let k = station_near(f, g.z);
    let zs = f.stations()[k];
    let mut out = Vec::new();
    for s in [1.0f64, -1.0] {
        let x0 = s * g.x;
        let lo = DVec3::new(x0 - 0.5 * g.leg, 0.0, zs - 0.5 * g.leg);
        let hi = DVec3::new(x0 + 0.5 * g.leg, f.bottom_y() - 0.03, zs + 0.5 * g.leg);
        let name = format!("landing_gear_{}", side_tag(s));
        let mut l = block(&name, "landing_gear", lo, hi, 0.5 * g.kg, 0.003, Mat::TrimBlack, vals(0.0, 0.0, g.legs.0, g.legs.1), Zone::Other);
        let tag = short_tag(&name);
        // its top four nodes (y index 1) on the rail's corners at the station and the next
        let from: Vec<String> = (0..4).map(|q| format!("{tag}{}1{}", q / 2, q % 2)).collect();
        let mut to = rail_nodes(f, s, &CORNERS, zs - 0.01 - f.spacing(), zs + 0.01 + f.spacing());
        to.extend(g.mount_to[usize::from(s < 0.0)].iter().cloned());
        mount(&mut l, &format!("{name}_brackets"), from, to, 16, vals(0.0, 0.0, g.brackets.0, g.brackets.1));
        // the fore-and-aft braces: the leg's foot to the rail's bottom a station ahead and
        // one behind (a leg held only at its top swings fore and aft as a cantilever)
        let feet: Vec<String> = (0..4).map(|q| format!("{tag}{}0{}", q / 2, q % 2)).collect();
        let mut aft = rail_nodes(f, s, &["ib", "ob"], zs - 0.01 - f.spacing(), zs - f.spacing() + 0.01);
        aft.extend(rail_nodes(f, s, &["ib", "ob"], zs + f.spacing() - 0.01, zs + 0.01 + f.spacing()));
        mount(&mut l, &format!("{name}_braces"), feet, aft, 4, vals(0.0, 0.0, g.brace.0, g.brace.1));
        l.mesh = leg_mesh(g, x0, zs, hi.y);
        out.push(l);
    }
    // the cross brace: each leg's inner nodes to the other's, top and bottom of the brace
    let (tl, tr) = (short_tag("landing_gear_L"), short_tag("landing_gear_R"));
    // (block nodes: {k x}{j y}{i z}; the left leg's inner side is k = 0, the right's k = 1)
    let mut links = Vec::new();
    for (j, i) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        for (j2, i2) in [(1 - j, i), (j, i)] {
            links.push((format!("{tl}0{j}{i}"), format!("{tr}1{j2}{i2}")));
        }
    }
    let l = out.first_mut().unwrap();
    for (a, b) in links {
        l.links.push(Link { name: format!("lg_brace_{a}_{b}"), a, b, v: vals(0.0, 0.0, g.brace.0, g.brace.1), group: None });
    }
    // the cross shaft between the legs and the crank on the left
    let mut shaft = box_(DVec3::new(0.0, 0.62, zs), DVec3::new(g.x - 0.5 * g.leg - 0.002, 0.02, 0.02), Mat::MetalBare, None);
    shaft.add(&box_(DVec3::new(g.x + 0.5 * g.leg + 0.12, 0.62, zs), DVec3::new(0.118, 0.012, 0.012), Mat::MetalBare, None));
    l.meshes.push("landing_gear_shaft".into());
    (out, vec![("landing_gear_shaft".into(), shaft)])
}

/// A leg's solid: its outer tube, the inner leg below it and the sand shoe on the ground.
fn leg_mesh(g: &LandingGear, x0: f64, zs: f64, top: f64) -> Mesh {
    let w = 0.5 * g.leg;
    let mut m = box_(DVec3::new(x0, 0.5 * (0.36 + top), zs), DVec3::new(w - 0.01, 0.5 * (top - 0.36), w - 0.01), Mat::TrimBlack, None);
    m.add(&box_(DVec3::new(x0, 0.2125, zs), DVec3::new(w - 0.03, 0.1425, w - 0.03), Mat::TrimBlack, None));
    m.add(&box_(DVec3::new(x0, 0.0375, zs), DVec3::new(w - 0.004, 0.0325, w - 0.004), Mat::TrimBlack, None));
    m
}
