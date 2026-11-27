//! The ladder frame: two C-channel side members and their cross-members.

use super::*;

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
        let mut r = part(&format!("frame_rail_{}", side_tag(s)), "frame", Zone::Other);
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
    let mut x = part("frame_crossmembers", "frame", Zone::Other);
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
