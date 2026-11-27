//! The cab: a three-station box on four mounts, and what rides it.

use super::*;

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
    let mut p = part("cab", "cab", Zone::Cabin);
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
    let extra = cab_meshes(c, paint);
    p.meshes = extra.iter().map(|e| e.0.clone()).collect();
    (p, extra)
}

/// What rides the cab: glass over black backings, the sun visor, grille, doors, handles,
/// mirrors, steps, the back wall's window and the roof's marker lamps.
fn cab_meshes(c: &Cab, paint: Mat) -> Vec<(String, Mesh)> {
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
    extra
}
