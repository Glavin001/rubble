//! Beam axles on spring hangers and their raycast wheels.

use super::*;

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
    let mut ax = part(name, "suspension", Zone::Wheel);
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
        let mut w = part(&format!("wheel_{corner}"), "wheel", Zone::Wheel);
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
