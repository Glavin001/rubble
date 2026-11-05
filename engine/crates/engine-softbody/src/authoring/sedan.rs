//! The showcase sedan: a 4.7 m mid-size four-door saloon (front engine, front
//! drive), 1,450 kg with its driver. Every number here is an editable design
//! input; `engine/docs/vehicle-authoring.md` walks through them.

use super::body::{BodyProfile, Well};
use super::curve::{Curve, Outline};
use super::material::Grades;
use super::members::{Corner, MemberSpec};
use super::vehicle::{
    BlockSpec, BraceSpec, Edge, FlangeSpec, Joint, Look, NamedMass, Panel, PlateSpec, Region,
    ShearSpec, SheetSpec, VehicleDef, Zone,
};

/// A member (left side when `mirrored`).
#[allow(clippy::too_many_arguments)]
pub fn member(
    name: &str,
    path: &[(f32, f32, f32)],
    width: f32,
    height: f32,
    up: [f32; 3],
    grade: &str,
    scale: [f32; 2],
    beads: usize,
    mirrored: bool,
    attach: usize,
) -> MemberSpec {
    MemberSpec {
        name: name.into(),
        path: path.iter().map(|&(x, y, z)| [x, y, z]).collect(),
        width,
        height,
        up,
        grade: grade.into(),
        scale,
        knee: None,
        beads,
        mirrored,
        attach,
        ride: false,
        section: 4,
        station: 0.0,
        catch: false,
        stiff: 1.0,
        skin: false,
        hardware_ends: true,
        kg_per_m: 0.0,
    }
}

/// A knee in a member's threshold gradient ([`MemberSpec::knee`]): scale `s`
/// at arc fraction `t`.
pub fn knee(mut m: MemberSpec, t: f32, s: f32) -> MemberSpec {
    m.knee = Some([t, s]);
    m
}

/// A member with its own station spacing.
pub fn stations(mut m: MemberSpec, spacing: f32) -> MemberSpec {
    m.station = spacing;
    m
}

/// A two-node (ladder) member.
pub fn ladder(mut m: MemberSpec) -> MemberSpec {
    m.section = 2;
    m
}

/// A box section `width` × `height` (cross members that carry mass: the dash
/// beam with the dashboard, the radiator support).
pub fn boxed(mut m: MemberSpec, width: f32, height: f32) -> MemberSpec {
    m.section = 4;
    m.width = width;
    m.height = height;
    m
}

/// A shear-panel diagonal (and its mirror).
pub fn brace(name: &str, from: [f32; 3], to: [f32; 3], grade: &str, kl: f32) -> BraceSpec {
    BraceSpec {
        name: name.into(),
        from,
        to,
        grade: grade.into(),
        kl,
        strength: 0.0,
        on: Default::default(),
    }
}

/// A brace whose ends lie on the members `from` and `to` (their left-side
/// names): explicit ends ([`BraceSpec::on`]).
pub fn on(mut b: BraceSpec, from: &str, to: &str) -> BraceSpec {
    b.on = [from.into(), to.into()];
    b
}

/// A brace that yields at `strength` N (into a crumple zone).
pub fn yielding(mut b: BraceSpec, strength: f32) -> BraceSpec {
    b.strength = strength;
    b
}

/// A heavier section (k·L × `s`).
pub fn stiff(mut m: MemberSpec, s: f32) -> MemberSpec {
    m.stiff = s;
    m
}

/// A passenger-cell member: its thresholds × [`CELL`] (the cell keeps its
/// shape in a 56 km/h crash; the crumple zones fold).
pub fn cell(mut m: MemberSpec) -> MemberSpec {
    m.scale = [m.scale[0] * CELL, m.scale[1] * CELL];
    m
}

/// The lumped windscreen panel's (`windscreen_x`) rigidity against
/// [`KL_PANEL`]: the glazing's measured share of the body's torsion (a
/// Choice of target within the published spread, `vehicle-authoring.md`).
pub const GLAZING: f32 = 2.0;

/// A lumped screen's (glass shear panel's) yield force (N): it cracks.
pub const GLASS_YIELD: f32 = 8.0e3;

/// A roof member (roof rails, bows, headers, B-pillars): its
/// thresholds × [`ROOF`] (the roof crushes in a roll-over; the A- and C-pillars,
/// sills and floor are the cell's).
pub fn roof(mut m: MemberSpec) -> MemberSpec {
    m.scale = [m.scale[0] * ROOF, m.scale[1] * ROOF];
    m
}

/// The roof members' threshold scale.
pub const ROOF: f32 = 1.2;

/// Yield force of the roof's lumped panels (N): with the roof members and
/// the B-pillars at their section's strength ([`side_sections`]), the roof
/// carries 5.1 × the unloaded weight within 127 mm of an FMVSS 216a platen
/// (compact sedans measure 4.48–5.32 ×; `crash-reference-targets.md`). At
/// 20 kN it carried 5.38 ×.
pub const ROOF_YIELD: f32 = 1.4e4;

/// Yield force of the cabin's door-ring panels (N): strong in a
/// 56 km/h crash, crushed by a roll-over.
pub const CABIN_YIELD: f32 = 6.0e4;

/// The hinge pillar's lumped body-side panel (the A-pillar's shoulder at the
/// cowl to the B-pillar's base): k·L × [`KL_PANEL`] and its yield force (N).
/// The yield is the panel's shear capacity: a 1.5 mm body-side outer and
/// hinge-pillar inner over about 0.6 m at about 200 MPa shear yield carry
/// about 180–200 kN. It keeps the front door's aperture within NCAP's
/// measured ±9 mm (`crash-reference-targets.md` §4.9) while the front crushes.
pub const HINGE_PANEL: f32 = 1.5;
pub const HINGE_YIELD: f32 = 2.0e5;

/// The doors' own frame (intrusion beam, belt and mid rails): thresholds ×
/// this against the boron grade. A ladder chord then yields at about 20 kN,
/// about 2 kN·m over its 0.1 m depth (a tubular intrusion beam's plastic
/// moment is a few kN·m): the door passes FMVSS 214's static crush (the `p`
/// test: ≥ 10 kN over 152 mm, ≥ 15.6 kN over 305 mm).
pub const DOOR_FRAME: f32 = 5.0;

/// The B-pillar's thresholds (× [`ROOF`]): it carries a side pole and the
/// door's latch end.
pub const PILLAR_B: f32 = 3.0;

/// The front subframe's cross members' threshold scale (a pole between the
/// engine and a rail meets them).
pub const SUBFRAME: f32 = 1.5;

/// The passenger cell's threshold scale.
pub const CELL: f32 = 4.0;

/// The firewall's lower cross member (the dash panel at the toe board)
/// against the boron grade. The rails and the engine push it back at 56 km/h
/// as NCAP measures (`crash-reference-targets.md` §1: left p10–p90 1–67 mm,
/// median 20; right −15–85): medians over 55–57 km/h 16 mm left, 0 right
/// (at 1.0 it stayed put, −1 mm). Weaker (0.2) and the front's weight on
/// the rail junctions yields it standing still.
pub const FIREWALL: f32 = 0.3;

/// An impact member: pole catchers between its stations.
pub fn catching(mut m: MemberSpec) -> MemberSpec {
    m.catch = true;
    m
}

/// A window frame (a door's sash): its ends carry no hardware.
pub fn window_frame(mut m: MemberSpec) -> MemberSpec {
    m.hardware_ends = false;
    m
}

/// A frame riding a panel (lid frames, door beams).
pub fn riding(mut m: MemberSpec) -> MemberSpec {
    m.ride = true;
    m
}

pub const UP: [f32; 3] = [0.0, 1.0, 0.0];
pub const OUT: [f32; 3] = [1.0, 0.0, 0.0];
pub const FWD: [f32; 3] = [0.0, 0.0, 1.0];

/// A cross member at station `z`, height `y`, from −x to x.
pub fn cross(name: &str, x: f32, y: f32, z: f32, grade: &str, scale: f32) -> MemberSpec {
    ladder(member(
        name,
        &[(-x, y, z), (x, y, z)],
        0.10,
        0.18,
        UP,
        grade,
        [scale, scale],
        0,
        false,
        2,
    ))
}

pub fn panel(name: &str, region: Region, grade: &str, joint: Joint, sided: bool) -> Panel {
    Panel {
        name: name.into(),
        region,
        grade: grade.into(),
        joint,
        sided,
    }
}

/// A seat frame: a light rigid lattice on breakable floor mounts.
pub fn seat_frame(name: &str, centre: [f32; 3], half: [f32; 3], kg: f32) -> BlockSpec {
    BlockSpec {
        name: name.into(),
        centre,
        half,
        spacing: 0.5,
        depth: 0.0,
        stiff: 1.0,
        kg,
        mounts: 3,
        reach: 0.45,
        mount_grade: "mount".into(),
        mount_scale: 3.0,
        bear: 0.0,
    }
}

pub fn door_joint() -> Joint {
    Joint::Hinged {
        hinge: Edge::Front,
        hinge_scale: 1.5,
        latch_scale: 1.2,
        latch_y: Some([0.5, 1.0]),
        latch_x: None,
    }
}

/// Yield force of the braces from the front strut towers into the cabin
/// (N): they fold with the crumple zone.
pub const END_YIELD: f32 = 2.0e4;

/// The front rails' threshold scale at their cabin end (under the floor at
/// the B-pillar) and at the bumper. Nearly the same along the rail (the
/// plateau starts at the bumper, so the pulse rises early, as measured); 5 %
/// weaker at the bumper so the force rises with the crush as Campbell's
/// model has it (`crash-reference-targets.md` §2.2): b1 is then one value at
/// 47–52 and 54–59 km/h (30.5 and 30.6 over 13 runs each; C̄(50)/C̄(56)
/// 0.877 against Campbell's 0.875). Uniform, b1 fell with speed (31.5 to
/// 31.0: 0.861); at 0.9 it rose (0.882), at 0.8 more (0.896).
pub const RAIL_CABIN: f32 = 1.0;
pub const RAIL_BUMPER: f32 = 0.7;

/// The crush grade's thresholds against `Grades::default`'s: the rails' and
/// aprons' plateau, set from NCAP measurements
/// (`engine/docs/crash-reference-targets.md` §2, §8): at 56 km/h a 1,450 kg
/// sedan crushes 0.62–0.75 m dynamically (t_m 63–76 ms) and 0.38–0.58 m
/// residually (Campbell b1 22.9–39.2). The `n` test measures it (1.0 since
/// the front axle moved 5 cm forward: its wheel meets the crush sooner).
pub const CRUSH_PLATEAU: f32 = 0.9;

/// The frame grades' (HSLA, UHSS, boron, aluminium, crush) rigidity against
/// `Grades::default`'s: the re-noded frame (about 330 nodes of 1–4 kg besides
/// the doors' frames, 2026-11-05) pays for half again as much at the 0.5 ms
/// step.
pub const FRAME_STIFF: f32 = 1.5;

/// The crush grade's rigidity against the frame's: a crushed front springs
/// back as NCAP measures. The seven 56 km/h sedans of
/// `crash-reference-targets.md` §8 spring back 0.10–0.24 m from their dynamic
/// crush (0.62–0.75 m) to their residual (§1: 0.42–0.57 m). The 56 km/h wall
/// gives dynamic 0.648 m, residual 0.520 m (medians over 55–57 km/h); at the
/// frame's rigidity the front sprang back 0.04 m and the dynamic crush fell
/// to 0.51–0.55 m.
pub const CRUSH_RIGIDITY: f32 = 0.15;

/// The powertrain lattice's stiffness against the rigid grade's: an engine
/// block is rigid, and its 175 kg pays for a lattice 3 × as stiff at the
/// step (it used 38 of its 72 kg of mass beyond its beams' need).
pub const POWERTRAIN_STIFF: f32 = 3.0;

/// The B-pillar's station spacing (m): two bays below the belt, so a side
/// pole at mid-span meets a station (the centre hinge of its 4·M_p/L
/// collapse); at 1 m the side pole's crush was 500 mm against 356.
pub const PILLAR_B_STATION: f32 = 0.6;

/// The doors' rails' station spacing (m): two bays along each (0.8 m), so a
/// pole meets a station mid-span (at 1 m the side pole's crush was 620 mm).
pub const DOOR_STATION: f32 = 0.42;

/// The rear body's upper chord: the quarter inner's and the C-pillar's
/// sections against their grades' k·L (the quarter inner and C-pillar carry
/// the rear's pitch into the roof and sill; with the rear rails into the
/// sills these cut the rear frame's motion in the cabin's frame under the
/// 56 km/h pulse from 107 mm to 23 mm, peak RMS).
pub const QUARTER_STIFF: f32 = 6.0;
pub const C_PILLAR_STIFF: f32 = 6.0;
/// The rear quarter's web (rear rail to quarter inner) against
/// [`KL_SHEAR`].
pub const QUARTER_WEB: f32 = 3.0;

/// The main load paths' (rails, sills) section against their grade's k·L.
pub const RAIL_STIFF: f32 = 1.4;

/// The rear rails' section against the HSLA grade's k·L.
pub const REAR_RAIL_STIFF: f32 = 3.5;

/// The rear rails' threshold scale from the kick-up to the rear subframe
/// (they carry the suspension, the fuel and the spare wheel: in a frontal
/// crash at 40 g that is ~40 kN a side), graded to 0.5 at the bumper behind
/// [`REAR_KNEE`] (the arc fraction of the rear subframe's station, z 0.62).
pub const REAR_RAIL: f32 = 4.0;
pub const REAR_KNEE: f32 = 0.73;

/// The rear subframe's cross tubes' k·L (N): see [`Corner::cross`].
pub const REAR_CROSS: f32 = 1.5e6;

/// The sills' section against the UHSS grade's k·L.
pub const SILL_STIFF: f32 = 2.3;

/// A lumped shear panel's axial rigidity (N): the cabin box's faces and the
/// strut towers' braces into it, spread over the sections they bear on.
pub const KL_PANEL: f32 = 7.0e6;

/// A zigzag shear panel's diagonals between two members (N).
pub const KL_SHEAR: f32 = 1.0e6;

/// The cabin box's lumped X braces (the shear panels' first form).
#[allow(clippy::too_many_arguments)]
fn cabin_x(
    fb: [f32; 3],
    ft: [f32; 3],
    cw: [f32; 3],
    bb: [f32; 3],
    bt: [f32; 3],
    rb: [f32; 3],
    rt: [f32; 3],
    k: f32,
) -> Vec<BraceSpec> {
    let m = |p: [f32; 3]| [-p[0], p[1], p[2]];
    vec![
        brace("floor_f", fb, m(bb), "floor", k),
        brace("floor_r", bb, m(rb), "floor", k),
        yielding(brace("roof_f", ft, m(bt), "panel", k), ROOF_YIELD),
        yielding(brace("roof_r", bt, m(rt), "panel", k), ROOF_YIELD),
        brace("firewall_x", fb, m(cw), "hsla", k),
        // The screen's frame (on the A-pillars' own stations), then its
        // glass (on the apron's and roof rail's).
        {
            let (kl, f) = screen_ring(cw, ft);
            on(
                yielding(brace("screen_ring", cw, m(ft), "uhss", kl), f),
                "pillar_a",
                "pillar_a",
            )
        },
        yielding(
            brace("windscreen_x", cw, m(ft), "glass", k * GLAZING),
            GLASS_YIELD,
        ),
        brace("bulkhead_x", rb, m(rt), "hsla", k),
    ]
    .into_iter()
    .chain(door_ring(
        "ring_f",
        [fb, ft, bb, bt],
        "pillar_a",
        "pillar_b",
    ))
    .chain(door_ring(
        "ring_r",
        [bb, bt, rb, rt],
        "pillar_b",
        "pillar_c",
    ))
    .collect()
}

/// A section from [`side_sections`] by its member's name.
fn section_of(name: &str) -> Section {
    side_sections()
        .into_iter()
        .find(|x| x.0 == name)
        .map(|x| x.1)
        .unwrap()
}

/// The same section bent in the other plane (depth and width swapped).
fn turned(s: Section) -> Section {
    Section {
        depth: s.width,
        width: s.depth,
        ..s
    }
}

/// A door ring (sill, front and rear pillars, roof rail) as the lumped X
/// of its frame in racking ([`aperture_racking`]) in the body side's
/// plane: the corners `[bottom front, top front, bottom rear, top rear]`,
/// the pillars `front` and `rear` bent fore and aft, the sill and roof
/// rail up and down ([`side_sections`]; the C-pillar's and A-pillar's
/// 100–120 mm fore-aft faces, the B-pillar's 120). Its two diagonals take
/// one k·L each, so their fore-aft stiffnesses sum to the frame's
/// (Σ k·L·cos²α/L), and yield together at the frame's yield shear
/// (Σ F·cos α). The columns sway in parallel (h/I over their mean I).
fn door_ring(name: &str, c: [[f32; 3]; 4], front: &str, rear: &str) -> Vec<BraceSpec> {
    let v = |p: [f32; 3]| glam::Vec3::from_array(p);
    let (bf, tf, br, tr) = (v(c[0]), v(c[1]), v(c[2]), v(c[3]));
    let (pf, pr) = (turned(section_of(front)), turned(section_of(rear)));
    // Columns in parallel: h/I with I the mean of the two.
    let i_col = 0.5 * (pf.i() + pr.i());
    let h = 0.5 * ((tf.y - bf.y) + (tr.y - br.y));
    let b = 0.5 * ((bf.z - br.z).abs() + (tf.z - tr.z).abs());
    let (top, bottom) = (section_of("roof_rail"), section_of("sill"));
    let flex =
        h * h / (24.0 * E_STEEL) * (h / i_col + 0.5 * b * (1.0 / top.i() + 1.0 / bottom.i()));
    let m_p = pf.m_p().min(pr.m_p()).min(top.m_p()).min(bottom.m_p());
    let (k, shear) = (1.0 / flex, 4.0 * m_p / h);
    let diags = [(c[0], c[3]), (c[2], c[1])];
    let geo: Vec<(f32, f32)> = diags
        .iter()
        .map(|&(a, b)| {
            let d = v(b) - v(a);
            (d.length(), d.z.abs() / d.length())
        })
        .collect();
    let kl = k / geo.iter().map(|&(l, cs)| cs * cs / l).sum::<f32>();
    let f = shear / geo.iter().map(|&(_, cs)| cs).sum::<f32>();
    vec![
        yielding(brace(&format!("{name}_x"), c[0], c[3], "uhss", kl), f),
        yielding(brace(&format!("{name}_y"), c[2], c[1], "uhss", kl), f),
    ]
}

/// The front axle (z, m): 2.75 m of wheelbase, 0.92 m of front overhang.
pub const FRONT_AXLE: f32 = 3.78;
/// The body's drop on its wheels (m): the wheels' centres stay at their
/// radius above the ground, everything else is this much lower.
pub const STANCE: f32 = 0.015;

/// The package the structure's points were drawn on (before the 2026-11-04
/// restyle: a roof 3 cm higher, the body 1.5 cm higher on its wheels, the
/// sides 1.5 cm narrower amidships): its top line and shoulder.
/// [`restyle_point`] carries a point drawn on it to the current body.
fn drawn_package() -> (Curve, Curve) {
    (
        Curve::new(&[
            (0.00, 0.99),
            (0.20, 1.02),
            (0.55, 1.045),
            (0.78, 1.06),
            (1.10, 1.25),
            (1.38, 1.395),
            (1.70, 1.435),
            (2.20, 1.445),
            (2.55, 1.425),
            (2.95, 1.22),
            (3.33, 1.00),
            (3.80, 0.945),
            (4.20, 0.875),
            (4.50, 0.825),
            (4.70, 0.765),
        ]),
        Curve::new(&[
            (0.00, 0.93),
            (0.40, 0.97),
            (0.78, 1.00),
            (1.40, 1.00),
            (2.30, 0.98),
            (3.00, 0.96),
            (3.50, 0.93),
            (3.90, 0.88),
            (4.30, 0.80),
            (4.70, 0.70),
        ]),
    )
}

/// A point of the structure drawn on [`drawn_package`], on the current body:
/// above the shoulder (the belt) it comes down with the roof in proportion
/// to its height between belt and roof; everything comes down by [`STANCE`].
/// Across it stays: the sides' extra 1.5 cm amidships is skin (widening the
/// members with it moved frame nodes off the skin nodes they shared, and the
/// short beams between them cost the skin a quarter of its stiffness).
fn restyle_point(p: [f32; 3], profile: &BodyProfile) -> [f32; 3] {
    let (top0, shoulder0) = drawn_package();
    let [x, y, z] = p;
    let (belt, roof0) = (shoulder0.eval(z), top0.eval(z));
    let drop = roof0 - (profile.top.eval(z) + STANCE);
    let share = if roof0 > belt {
        ((y - belt) / (roof0 - belt)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    [x, y - drop * share - STANCE, z]
}

/// A side-view point (z, y) drawn on [`drawn_package`], on the current body.
fn restyle_side(q: [f32; 2], profile: &BodyProfile) -> [f32; 2] {
    let p = restyle_point([0.0, q[1], q[0]], profile);
    [q[0], p[1]]
}

fn restyle_region(r: &mut Region, profile: &BodyProfile) {
    match r {
        Region::Side(o) => {
            for q in &mut o.0 {
                *q = restyle_side(*q, profile);
            }
        }
        Region::Cap { y0, y1, .. } => {
            *y0 -= STANCE;
            *y1 -= STANCE;
        }
        Region::Box { x, y, z } => {
            let a = restyle_point([x[0], y[0], z[0]], profile);
            let b = restyle_point([x[1], y[1], z[1]], profile);
            (*x, *y) = ([a[0], b[0]], [a[1], b[1]]);
        }
        Region::Any(v) => {
            for r in v {
                restyle_region(r, profile);
            }
        }
        Region::Top { .. } | Region::Floor { .. } => {}
    }
}

/// Carries every point drawn on [`drawn_package`] (members, braces, blocks,
/// plates, named masses, panel and zone regions, lamps, handles, mirror and
/// plates) to the current body. The side glass, wheels and wells are drawn
/// on the current body already.
fn restyle(mut d: VehicleDef) -> VehicleDef {
    let pr = d.profile.clone();
    let pt = |p: &mut [f32; 3]| *p = restyle_point(*p, &pr);
    for m in &mut d.members {
        m.path.iter_mut().for_each(pt);
    }
    for b in &mut d.braces {
        pt(&mut b.from);
        pt(&mut b.to);
    }
    for b in &mut d.blocks {
        pt(&mut b.centre);
    }
    for p in &mut d.plates {
        pt(&mut p.corner);
    }
    for m in &mut d.masses {
        pt(&mut m.lo);
        pt(&mut m.hi);
    }
    for p in &mut d.panels {
        restyle_region(&mut p.region, &pr);
    }
    for z in &mut d.zones {
        restyle_region(&mut z.region, &pr);
    }
    let l = &mut d.look;
    for o in [
        &mut l.headlight,
        &mut l.grille,
        &mut l.intake,
        &mut l.taillight,
    ] {
        for q in &mut o.0 {
            q[1] -= STANCE;
        }
    }
    for h in &mut l.handles {
        *h = restyle_side(*h, &pr);
    }
    l.mirror = restyle_side(l.mirror, &pr);
    l.plate_y[0] -= STANCE;
    d
}

/// A thin-walled closed section (a box, or a hat closed by its panel) of a
/// member, for its lateral plastic collapse: `depth` along the load (the
/// lever), `width` across it, wall `t` (m), the steel's yield stress
/// `sigma_y` (Pa). A round tube (`round`) has diameter `depth`.
#[derive(Clone, Copy, Debug)]
pub struct Section {
    pub depth: f32,
    pub width: f32,
    pub t: f32,
    pub sigma_y: f32,
    pub round: bool,
}

impl Section {
    /// Plastic section modulus (m³): a box's two flanges (t·width at
    /// ±depth/2) and two webs (t·depth²/4 each); a tube's
    /// (D³ − (D − 2t)³)/6.
    pub fn z_p(&self) -> f32 {
        if self.round {
            let (d, i) = (self.depth, self.depth - 2.0 * self.t);
            (d * d * d - i * i * i) / 6.0
        } else {
            self.t * (self.width * self.depth + 0.5 * self.depth * self.depth)
        }
    }

    /// Plastic moment M_p = σ_y·Z_p (N·m).
    pub fn m_p(&self) -> f32 {
        self.sigma_y * self.z_p()
    }

    /// Second moment of area about the axis across `depth` (m⁴): a thin
    /// box's t·d²(3w + d)/6, a thin tube's π·r³·t.
    pub fn i(&self) -> f32 {
        if self.round {
            let r = 0.5 * (self.depth - self.t);
            std::f32::consts::PI * r * r * r * self.t
        } else {
            self.t * self.depth * self.depth * (3.0 * self.width + self.depth) / 6.0
        }
    }
}

/// Steel's Young's modulus (Pa).
pub const E_STEEL: f32 = 210e9;

/// A window aperture's frame in racking (exact for a closed rectangular
/// frame with rigid corners, each member in double curvature, its
/// inflection at mid-span): pillars of length `h` and second moment
/// `pillar.i()`, the top and bottom members `b` long. The racking
/// displacement under a shear V is V·h²/(24E)·[h/I_p + b/2·(1/I_t + 1/I_b)],
/// and the frame yields when its weakest corner member reaches M_p: the
/// corner moment is V·h/4. Returns (stiffness N/m, yield shear N).
pub fn aperture_racking(
    pillar: Section,
    top: Section,
    bottom: Section,
    h: f32,
    b: f32,
) -> (f32, f32) {
    let flex =
        h * h / (24.0 * E_STEEL) * (h / pillar.i() + 0.5 * b * (1.0 / top.i() + 1.0 / bottom.i()));
    let m_p = pillar.m_p().min(top.m_p()).min(bottom.m_p());
    (1.0 / flex, 4.0 * m_p / h)
}

/// The windscreen aperture's frame (A-pillars, front header, cowl) as a
/// lumped X from the cowl's corner to the header's opposite corner (and its
/// mirror), of the frame's racking stiffness and yield
/// ([`aperture_racking`]); the sections in the screen's plane: the
/// A-pillar's ([`side_sections`]) bent across its 100 mm, the header's
/// across its 100 mm fore-aft, the cowl's plenum a 150 × 100 × 1.0 mm DP600
/// box. Rigid corners are a Choice (a real joint is softer). Returns
/// (k·L of each diagonal, its yield force).
pub fn screen_ring(cw: [f32; 3], ft: [f32; 3]) -> (f32, f32) {
    let sec = |depth: f32, width: f32, t: f32, sigma_y: f32| Section {
        depth,
        width,
        t,
        sigma_y,
        round: false,
    };
    let pillar = sec(0.10, 0.12, 0.0014, DP780);
    let header = sec(0.10, 0.06, 0.0012, HSLA340);
    let cowl = sec(0.15, 0.10, 0.0010, DP600);
    let (a, c) = (glam::Vec3::from_array(cw), glam::Vec3::from_array(ft));
    // A pillar from the cowl's corner to the header's; the frame's width
    // the mean of its top and bottom.
    let h = a.distance(c);
    let b = a.x + c.x;
    let (k, v) = aperture_racking(pillar, header, cowl, h, b);
    // Each diagonal (cowl corner to the opposite header corner) carries
    // half the shear along x: k = 2·(kL/L)·cos²α, V = 2·F·cos α.
    let d = glam::Vec3::new(a.x + c.x, a.y - c.y, a.z - c.z);
    let (l, cos) = (d.length(), (a.x + c.x) / d.length());
    (k * l / (2.0 * cos * cos), v / (2.0 * cos))
}

/// Grades' minimum yield stresses (Pa): EN 10338 for the dual-phase steels
/// (DP600 330, DP780 440, DP980 590), EN 10268 for HSLA CR340LA,
/// EN 10149-2 for S420MC, about 1,100 for hot-stamped 22MnB5 and 1,200
/// for an ultra-high-strength door-beam tube, EN 755-2 for an AW-6082-T6
/// extrusion (250). The same grades and classes as the 2011 Honda Accord
/// teardown's material table (NHTSA DOT HS 811 666, fig. 113: DP 350/600,
/// DP 500/800, HSLA 350/450, HF 1050/1500).
pub const DP600: f32 = 330e6;
/// DP600's minimum tensile strength (Pa), EN 10338 (HCT600X).
pub const DP600_UTS: f32 = 600e6;
pub const DP780: f32 = 440e6;
pub const DP980: f32 = 590e6;
pub const HSLA340: f32 = 340e6;
pub const S420: f32 = 420e6;
pub const PHS: f32 = 1100e6;
pub const TUBE1200: f32 = 1200e6;
pub const AL6082: f32 = 250e6;

/// The members' real sections and the truss's lever for their bending:
/// `(name, section, lever (m), chords a side)`. A box truss carries
/// M = 2·T·lever, a ladder M = T·lever, so each member's chords yield at
/// T = M_p / (chords · lever) and the truss's plastic moment is the
/// section's. (End-on, a cross member's two chords then carry 2·M_p/depth,
/// within a third of its section's squash load A·σ_y.) Sections are typical
/// of compact and mid-size sedan bodies (a Choice: not one car's drawings);
/// `vehicle-authoring.md` (Lateral strength) tabulates them.
pub fn side_sections() -> Vec<(&'static str, Section, f32, f32)> {
    let sec = |depth: f32, width: f32, t: f32, sigma_y: f32| Section {
        depth,
        width,
        t,
        sigma_y,
        round: false,
    };
    let tube = |d: f32, t: f32, sigma_y: f32| Section {
        depth: d,
        width: d,
        t,
        sigma_y,
        round: true,
    };
    vec![
        // Rocker: inner, outer and reinforcement, 100 wide × 140 tall, 1.8 mm.
        ("sill", sec(0.10, 0.14, 0.0018, DP980), 0.18, 2.0),
        // B-pillar: hot-stamped hat closed by its outer, 70 deep × 120 wide.
        ("pillar_b", sec(0.07, 0.12, 0.0018, PHS), 0.14, 2.0),
        // Tunnel: hat 200 wide × 150 tall closed by the floor, 1.2 mm.
        ("tunnel", sec(0.20, 0.15, 0.0012, DP600), 0.16, 2.0),
        // Front seat cross members: hats 80 × 60 closed by the floor.
        ("floor_x1", sec(0.08, 0.06, 0.0014, DP780), 0.08, 1.0),
        ("floor_x2", sec(0.08, 0.06, 0.0016, DP780), 0.08, 1.0),
        // Rear seat and kick-up cross members: hats 100 × 60, 1.2 mm.
        ("floor_x3", sec(0.10, 0.06, 0.0012, HSLA340), 0.10, 1.0),
        ("kickup", sec(0.10, 0.06, 0.0012, HSLA340), 0.10, 1.0),
        // Roof side rails: inner, outer and reinforcement, 70 along the
        // load × 90, 1.4 mm.
        ("roof_rail", sec(0.07, 0.09, 0.0014, DP780), 0.10, 1.0),
        // Front and rear headers and the B-pillar's roof bow: hats 60 × 100
        // closed by the roof, 1.2 mm.
        ("header_f", sec(0.06, 0.10, 0.0012, HSLA340), 0.18, 1.0),
        ("header_r", sec(0.06, 0.10, 0.0012, HSLA340), 0.18, 1.0),
        ("bow_b", sec(0.06, 0.10, 0.0012, HSLA340), 0.18, 1.0),
        // The rear roof bow: a 40 × 80 hat, 0.8 mm.
        ("bow_r", sec(0.04, 0.08, 0.0008, HSLA340), 0.18, 1.0),
        // A-pillar (hinge pillar into the screen pillar): inner, outer and
        // reinforcement, 100 across the car × 120, 1.4 mm.
        ("pillar_a", sec(0.10, 0.12, 0.0014, DP780), 0.10, 2.0),
        // C-pillar: 70 across the car × 100, 1.4 mm.
        ("pillar_c", sec(0.07, 0.10, 0.0014, DP780), 0.12, 1.0),
        // Rear rails: 100 across × 130 tall, 1.6 mm.
        ("rail_r", sec(0.10, 0.13, 0.0016, DP600), 0.10, 2.0),
        ("rail_r_tail", sec(0.10, 0.13, 0.0016, DP600), 0.10, 2.0),
        // Front and rear subframe cross members (and the front subframe's
        // rear one): 90 × 60 boxes, 2.0 mm.
        ("subframe_f0", sec(0.09, 0.06, 0.0020, S420), 0.18, 1.0),
        ("subframe_f1", sec(0.09, 0.06, 0.0020, S420), 0.18, 1.0),
        ("subframe_f2", sec(0.09, 0.06, 0.0020, S420), 0.18, 1.0),
        ("subframe_r1", sec(0.09, 0.06, 0.0020, S420), 0.18, 1.0),
        ("subframe_r2", sec(0.09, 0.06, 0.0020, S420), 0.18, 1.0),
        // Rear seat cushion cross member: an 80 × 60 hat, 1.2 mm.
        ("seat_x", sec(0.08, 0.06, 0.0012, DP600), 0.08, 1.0),
        // Rear bumper beam: an AW-6082-T6 extrusion 50 deep × 100, 2.5 mm.
        ("bumper_beam_r", sec(0.05, 0.10, 0.0025, AL6082), 0.12, 1.0),
        // Instrument panel cross-car beam: a steel tube Ø 60 × 1.6 mm.
        ("dash", tube(0.060, 0.0016, S420), 0.18, 1.0),
        // Door intrusion beams: tubes Ø 31.8 × 1.9 mm, 1,200 MPa.
        ("door_beam_f", tube(0.0318, 0.0019, TUBE1200), 0.14, 1.0),
        ("door_beam_r", tube(0.0318, 0.0019, TUBE1200), 0.14, 1.0),
    ]
}

/// Sets each [`side_sections`] member's thresholds from its section's
/// plastic moment (a member with a knee keeps its gradient's shape, its
/// strongest point at the section's).
fn sectioned(members: &mut [MemberSpec], grades: &Grades) {
    sectioned_by(members, grades, &side_sections());
}

/// [`sectioned`] for a list of (member, section, lever, sides).
fn sectioned_by(
    members: &mut [MemberSpec],
    grades: &Grades,
    list: &[(&'static str, Section, f32, f32)],
) {
    for &(name, sec, lever, sides) in list {
        for m in members.iter_mut().filter(|m| m.name == name) {
            let chord = sec.m_p() / (sides * lever);
            let s = chord / grades.get(&m.grade).unwrap().deform;
            match m.knee {
                Some([t, k]) => {
                    let top = m.scale[0].max(m.scale[1]).max(k);
                    let f = s / top;
                    m.scale = [m.scale[0] * f, m.scale[1] * f];
                    m.knee = Some([t, k * f]);
                }
                None => m.scale = [s, s],
            }
        }
    }
}

pub fn sedan() -> VehicleDef {
    let length = 4.70;
    // A greenhouse 3 cm lower than the package the structure was drawn on
    // (the belt unchanged), the body 1.5 cm lower on its wheels (`STANCE`),
    // the sides 1.5 cm wider amidships, the front axle 5 cm forward, the
    // bonnet's leading edge 3 cm lower.
    let profile = BodyProfile {
        length,
        top: Curve::new(&[
            (0.00, 0.975),
            (0.20, 1.005),
            (0.55, 1.03),
            (0.78, 1.045),
            (1.10, 1.21),
            (1.38, 1.35),
            (1.70, 1.39),
            (2.20, 1.40),
            (2.55, 1.38),
            (2.95, 1.185),
            (3.33, 0.985),
            (3.80, 0.93),
            (4.20, 0.86),
            (4.50, 0.79),
            (4.70, 0.72),
        ]),
        bottom: Curve::new(&[
            (0.00, 0.405),
            (0.25, 0.285),
            (0.55, 0.205),
            (0.90, 0.165),
            (1.20, 0.155),
            (3.60, 0.155),
            (4.15, 0.175),
            (4.45, 0.215),
            (4.70, 0.295),
        ]),
        shoulder: Curve::new(&[
            (0.00, 0.915),
            (0.40, 0.955),
            (0.78, 0.985),
            (1.40, 0.985),
            (2.30, 0.965),
            (3.00, 0.945),
            (3.50, 0.915),
            (3.90, 0.865),
            (4.30, 0.785),
            (4.70, 0.685),
        ]),
        half_width: Curve::new(&[
            (0.00, 0.80),
            (0.25, 0.86),
            (0.60, 0.895),
            (1.20, 0.92),
            (2.40, 0.925),
            (3.60, 0.92),
            (4.20, 0.88),
            (4.50, 0.84),
            (4.70, 0.78),
        ]),
        top_half_width: Curve::new(&[
            (0.00, 0.62),
            (0.40, 0.70),
            (0.78, 0.71),
            (1.38, 0.66),
            (2.00, 0.645),
            (2.55, 0.66),
            (3.35, 0.75),
            (4.00, 0.76),
            (4.50, 0.70),
            (4.70, 0.60),
        ]),
        crown: 0.04,
        side_bulge: 0.015,
        lower_radius: 0.07,
        top_radius: 0.05,
        front_sweep: 0.20,
        rear_sweep: 0.14,
        wells: vec![
            Well {
                z: 1.03,
                y: 0.335,
                radius: 0.385,
                inner_x: 0.58,
            },
            Well {
                z: FRONT_AXLE,
                y: 0.335,
                radius: 0.385,
                inner_x: 0.58,
            },
        ],
    };
    let side_glass = vec![
        Outline::new(&[(3.09, 0.995), (2.65, 1.31), (2.36, 1.33), (2.36, 0.995)]),
        Outline::new(&[(2.12, 0.995), (2.12, 1.33), (1.64, 1.30), (1.47, 1.005)]),
    ];
    let panels = vec![
        panel(
            "bumper_f",
            Region::Any(vec![
                Region::Cap {
                    front: true,
                    y0: 0.0,
                    y1: 0.70,
                },
                Region::Box {
                    x: [0.0, 1.0],
                    y: [0.0, 0.60],
                    z: [4.18, 4.8],
                },
            ]),
            "plastic",
            Joint::Bolt {
                scale: 1.0,
                group: false,
            },
            false,
        ),
        panel(
            "bumper_r",
            Region::Any(vec![
                Region::Cap {
                    front: false,
                    y0: 0.0,
                    y1: 0.62,
                },
                Region::Box {
                    x: [0.0, 1.0],
                    y: [0.0, 0.58],
                    z: [-0.1, 0.55],
                },
            ]),
            "plastic",
            Joint::Bolt {
                scale: 0.8,
                group: false,
            },
            false,
        ),
        panel(
            "bonnet",
            Region::Any(vec![
                Region::Top { z0: 3.55, z1: 4.8 },
                Region::Cap {
                    front: true,
                    y0: 0.70,
                    y1: 2.0,
                },
            ]),
            "panel",
            Joint::Hinged {
                hinge: Edge::Rear,
                hinge_scale: 1.0,
                latch_scale: 0.8,
                latch_y: None,
                latch_x: Some(0.3),
            },
            false,
        ),
        panel(
            "bootlid",
            Region::Any(vec![
                Region::Top { z0: -0.1, z1: 0.76 },
                Region::Cap {
                    front: false,
                    y0: 0.80,
                    y1: 2.0,
                },
            ]),
            "panel",
            Joint::Hinged {
                hinge: Edge::Front,
                hinge_scale: 1.0,
                latch_scale: 0.8,
                latch_y: None,
                latch_x: Some(0.3),
            },
            false,
        ),
        panel(
            "fender",
            Region::Side(Outline::rect(3.36, 0.2, 4.8, 1.2)),
            "panel",
            // Bolted along its top and to the A-pillar: holds at 56 km/h
            // (the fender crumples back), tears at about 80 and up.
            Joint::Bolt {
                scale: 2.0,
                group: false,
            },
            true,
        ),
        panel(
            "door_f",
            Region::Side(Outline::new(&[
                (3.27, 0.29),
                (3.27, 0.99),
                (2.61, 1.40),
                (2.30, 1.42),
                (2.30, 0.29),
            ])),
            "panel",
            door_joint(),
            true,
        ),
        panel(
            "door_r",
            Region::Side(Outline::new(&[
                (2.18, 0.29),
                (2.18, 1.42),
                (1.62, 1.38),
                (1.40, 1.02),
                (1.40, 0.29),
            ])),
            "panel",
            door_joint(),
            true,
        ),
    ];
    let mut zones = vec![
        Zone {
            region: Region::Top { z0: 2.59, z1: 3.32 },
            grade: "glass".into(),
        },
        Zone {
            region: Region::Top { z0: 0.81, z1: 1.35 },
            grade: "glass".into(),
        },
    ];
    for o in &side_glass {
        zones.push(Zone {
            region: Region::Side(o.clone()),
            grade: "glass".into(),
        });
    }
    zones.extend([Zone {
        region: Region::Floor { z0: -1.0, z1: 6.0 },
        grade: "floor".into(),
    }]);
    // The frame: a space frame of members that meet at shared joint nodes
    // (every path vertex is a station; a station within `members::SNAP` of
    // another member's frame node is that node). The cabin's corners: the A-,
    // B- and C-pillar bases on the sills, their tops on the roof rails; the
    // floor's cross members meet the sills, the rails (x 0.44) and the tunnel.
    let rs = RAIL_STIFF;
    let floor_cross = |name: &str, z: f32, grade: &str| -> MemberSpec {
        ladder(member(
            name,
            &[
                (-0.79, 0.27, z),
                (-0.44, 0.26, z),
                (0.0, 0.30, z),
                (0.44, 0.26, z),
                (0.79, 0.27, z),
            ],
            0.10,
            0.16,
            UP,
            grade,
            [1.0, 1.0],
            0,
            false,
            2,
        ))
    };
    let members = vec![
        // Front crash rails: under the floor from the B-pillar's cross member,
        // up the firewall, under the strut tower, to the bumper beam; graded
        // strong (cabin) to weak (crash boxes), beaded.
        stiff(
            member(
                "rail_f",
                &[
                    (0.44, 0.23, 2.24),
                    (0.44, 0.24, 2.80),
                    (0.44, 0.31, 3.36),
                    (0.44, 0.41, FRONT_AXLE),
                    (0.44, 0.42, 4.10),
                    (0.44, 0.43, 4.40),
                    (0.44, 0.43, 4.57),
                ],
                0.14,
                0.24,
                UP,
                "crush",
                [RAIL_CABIN, RAIL_BUMPER],
                3,
                true,
                1,
            ),
            rs,
        ),
        // Upper rails (aprons) from the A-pillar at the cowl over the strut
        // tower to the radiator support.
        ladder(member(
            "apron",
            &[
                (0.79, 0.92, 3.38),
                (0.66, 0.82, FRONT_AXLE),
                (0.55, 0.74, 4.40),
            ],
            0.14,
            0.14,
            UP,
            "crush",
            [1.4, 0.5],
            3,
            true,
            1,
        )),
        // Strut towers: the rail to the apron.
        ladder(member(
            "tower_f",
            &[(0.44, 0.41, FRONT_AXLE), (0.66, 0.82, FRONT_AXLE)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [1.5, 1.5],
            0,
            true,
            1,
        )),
        ladder(member(
            "tower_r",
            &[(0.45, 0.40, 1.03), (0.68, 0.86, 1.03)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [1.5, 1.5],
            0,
            true,
            1,
        )),
        catching(stiff(
            cell(member(
                "sill",
                &[
                    (0.79, 0.31, 1.36),
                    (0.79, 0.31, 1.80),
                    (0.79, 0.31, 2.24),
                    (0.79, 0.31, 2.80),
                    (0.79, 0.31, 3.36),
                ],
                0.18,
                0.24,
                UP,
                "uhss",
                [1.0, 1.0],
                0,
                true,
                2,
            )),
            SILL_STIFF,
        )),
        cell(member(
            "pillar_a",
            &[(0.79, 0.31, 3.36), (0.79, 0.92, 3.38), (0.62, 1.37, 2.59)],
            0.10,
            0.12,
            FWD,
            "uhss",
            [1.2, 1.2],
            0,
            true,
            1,
        )),
        catching(stations(
            roof(member(
                "pillar_b",
                &[(0.79, 0.31, 2.24), (0.79, 0.98, 2.24), (0.62, 1.39, 2.24)],
                0.14,
                0.14,
                OUT,
                "boron",
                [PILLAR_B, PILLAR_B],
                0,
                true,
                1,
            )),
            PILLAR_B_STATION,
        )),
        stiff(
            ladder(cell(member(
                "pillar_c",
                &[(0.79, 0.31, 1.36), (0.79, 0.98, 1.36), (0.62, 1.37, 1.50)],
                0.10,
                0.12,
                FWD,
                "uhss",
                [1.0, 1.0],
                0,
                true,
                1,
            ))),
            C_PILLAR_STIFF,
        ),
        ladder(roof(member(
            "roof_rail",
            &[
                (0.62, 1.37, 2.59),
                (0.62, 1.39, 2.24),
                (0.62, 1.39, 1.79),
                (0.62, 1.37, 1.50),
            ],
            0.10,
            0.10,
            UP,
            "uhss",
            [1.0, 1.0],
            0,
            true,
            1,
        ))),
        roof(cross("header_f", 0.62, 1.37, 2.59, "uhss", 1.0)),
        roof(cross("bow_b", 0.62, 1.39, 2.24, "uhss", 1.0)),
        // A rear roof bow (the roof skin's span between bows sags otherwise)
        // under a ring of the roof skin, with a station under each of its
        // nodes (x 0, ±0.31): the skin's flat centre otherwise snaps through
        // (it inverts without straining a beam) and stays 2 cm down after a
        // frontal crash.
        stations(roof(cross("bow_r", 0.62, 1.39, 1.79, "uhss", 1.0)), 0.31),
        roof(cross("header_r", 0.62, 1.37, 1.50, "uhss", 1.0)),
        catching(cell(floor_cross("floor_x1", 2.80, "uhss"))),
        catching(cell(floor_cross("floor_x2", 2.24, "boron"))),
        cell(floor_cross("floor_x3", 1.80, "uhss")),
        cell(floor_cross("kickup", 1.36, "uhss")),
        catching(cell(member(
            "tunnel",
            &[
                (0.0, 0.33, 1.36),
                (0.0, 0.33, 1.80),
                (0.0, 0.33, 2.24),
                (0.0, 0.33, 2.80),
                (0.0, 0.33, 3.36),
            ],
            0.16,
            0.20,
            UP,
            "boron",
            [1.0, 1.0],
            0,
            false,
            1,
        ))),
        catching({
            let mut m = floor_cross("firewall", 3.36, "boron");
            m.scale = [FIREWALL, FIREWALL];
            m
        }),
        ladder(cell(member(
            "dash",
            &[
                (-0.79, 0.80, 3.15),
                (-0.38, 0.80, 3.09),
                (0.38, 0.80, 3.09),
                (0.79, 0.80, 3.15),
            ],
            0.10,
            0.18,
            UP,
            "uhss",
            [1.0, 1.0],
            0,
            false,
            2,
        ))),
        ladder(cell(member(
            "cowl",
            &[
                (-0.79, 0.92, 3.38),
                (-0.40, 0.92, 3.29),
                (0.40, 0.92, 3.29),
                (0.79, 0.92, 3.38),
            ],
            0.10,
            0.18,
            UP,
            "uhss",
            [1.0, 1.0],
            0,
            false,
            2,
        ))),
        ladder(member(
            "rad_post",
            &[(0.44, 0.43, 4.40), (0.55, 0.74, 4.40)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [0.6, 0.6],
            0,
            true,
            1,
        )),
        cross("rad_top", 0.55, 0.74, 4.40, "hsla", 0.6),
        catching(ladder(member(
            "bumper_beam_f",
            &[
                (-0.72, 0.45, 4.49),
                (-0.44, 0.45, 4.57),
                (0.0, 0.45, 4.63),
                (0.44, 0.45, 4.57),
                (0.72, 0.45, 4.49),
            ],
            0.10,
            0.12,
            FWD,
            "uhss",
            [1.0, 1.0],
            0,
            false,
            1,
        ))),
        catching(ladder(member(
            "subframe_f1",
            &[
                (-0.44, 0.29, FRONT_AXLE),
                (0.0, 0.27, FRONT_AXLE),
                (0.44, 0.29, FRONT_AXLE),
            ],
            0.10,
            0.18,
            UP,
            "hsla",
            [SUBFRAME, SUBFRAME],
            0,
            false,
            2,
        ))),
        catching(ladder(member(
            "subframe_f2",
            &[(-0.44, 0.30, 4.10), (0.0, 0.28, 4.10), (0.44, 0.30, 4.10)],
            0.10,
            0.18,
            UP,
            "hsla",
            [SUBFRAME, SUBFRAME],
            0,
            false,
            2,
        ))),
        // The steering column, from the dash cross beam (a box: it is a
        // cantilever).
        member(
            "column",
            &[(0.38, 0.80, 3.09), (0.38, 0.93, 2.83)],
            0.10,
            0.10,
            UP,
            "hsla",
            [1.0, 1.0],
            0,
            false,
            0,
        ),
        // Rear rails from the rear seat's cross member over the kick-up,
        // under the rear strut tower, to the rear subframe (z 0.62); their
        // tail on to the rear bumper beam is `rail_r_tail`.
        stiff(
            member(
                "rail_r",
                &[
                    (0.70, 0.26, 2.24),
                    (0.44, 0.22, 1.80),
                    (0.44, 0.24, 1.36),
                    (0.44, 0.40, 1.03),
                    (0.44, 0.42, 0.62),
                ],
                0.10,
                0.13,
                UP,
                "hsla",
                [REAR_RAIL, REAR_RAIL],
                3,
                true,
                1,
            ),
            REAR_RAIL_STIFF,
        ),
        // The rear rails' tail behind the rear subframe (the rear crush
        // zone), graded from the knee's strength to 0.5 at the bumper. A
        // real rail is continuous; static contact is per node, so its
        // stations are at most a 254 mm NCAP pole's width apart (z 0.62,
        // 0.40, 0.17): a pole in any lane meets one. Its k·L is the HSLA
        // grade's (a Choice: at ×3.5 its two extra stations cost the step
        // budget 32 kg of stability floor).
        knee(
            member(
                "rail_r_tail",
                &[(0.44, 0.42, 0.62), (0.44, 0.435, 0.40), (0.44, 0.45, 0.17)],
                0.10,
                0.13,
                UP,
                "hsla",
                [REAR_RAIL, 0.5],
                0,
                true,
                1,
            ),
            0.0,
            REAR_RAIL,
        ),
        catching(ladder(member(
            "bumper_beam_r",
            &[
                (-0.70, 0.48, 0.21),
                (-0.44, 0.48, 0.15),
                (0.0, 0.48, 0.11),
                (0.44, 0.48, 0.15),
                (0.70, 0.48, 0.21),
            ],
            0.10,
            0.12,
            FWD,
            "alu",
            [1.0, 1.0],
            0,
            false,
            1,
        ))),
        // Rear aprons (inner quarter panels) from the C-pillar over the rear
        // strut tower.
        stiff(
            ladder(member(
                "quarter_inner",
                &[(0.79, 0.98, 1.36), (0.68, 0.86, 1.03), (0.66, 0.88, 0.24)],
                0.12,
                0.12,
                UP,
                "hsla",
                [1.4, 0.6],
                0,
                true,
                1,
            )),
            QUARTER_STIFF,
        ),
        ladder(member(
            "subframe_r1",
            &[(-0.44, 0.31, 1.03), (0.0, 0.29, 1.03), (0.44, 0.31, 1.03)],
            0.10,
            0.18,
            UP,
            "hsla",
            [1.5, 1.5],
            0,
            false,
            2,
        )),
        ladder(member(
            "subframe_r2",
            &[(-0.44, 0.33, 0.62), (0.0, 0.31, 0.62), (0.44, 0.33, 0.62)],
            0.10,
            0.18,
            UP,
            "hsla",
            [1.5, 1.5],
            0,
            false,
            2,
        )),
        ladder(cell(member(
            "parcel",
            &[
                (-0.79, 0.98, 1.36),
                (-0.44, 1.00, 1.36),
                (0.44, 1.00, 1.36),
                (0.79, 0.98, 1.36),
            ],
            0.10,
            0.18,
            UP,
            "hsla",
            [1.0, 1.0],
            0,
            false,
            1,
        ))),
        // Door intrusion beams (they ride their door).
        catching(stations(
            riding(ladder(member(
                "door_beam_f",
                &[(0.83, 0.42, 3.18), (0.83, 0.62, 2.40)],
                0.10,
                0.14,
                OUT,
                "boron",
                [DOOR_FRAME, DOOR_FRAME],
                0,
                true,
                3,
            ))),
            DOOR_STATION,
        )),
        // Belt reinforcements under the windows (they ride their door).
        stations(
            riding(ladder(member(
                "door_belt_f",
                &[(0.84, 0.92, 3.20), (0.84, 0.94, 2.36)],
                0.10,
                0.14,
                OUT,
                "boron",
                [DOOR_FRAME, DOOR_FRAME],
                0,
                true,
                3,
            ))),
            DOOR_STATION,
        ),
        stations(
            riding(ladder(member(
                "door_belt_r",
                &[(0.84, 0.94, 2.12), (0.84, 0.94, 1.48)],
                0.10,
                0.14,
                OUT,
                "boron",
                [DOOR_FRAME, DOOR_FRAME],
                0,
                true,
                3,
            ))),
            DOOR_STATION,
        ),
        // Inner rails at mid height (they ride their door).
        stations(
            riding(ladder(member(
                "door_mid_f",
                &[(0.83, 0.74, 3.22), (0.83, 0.76, 2.36)],
                0.10,
                0.14,
                OUT,
                "boron",
                [DOOR_FRAME, DOOR_FRAME],
                0,
                true,
                3,
            ))),
            DOOR_STATION,
        ),
        stations(
            riding(ladder(member(
                "door_mid_r",
                &[(0.83, 0.76, 2.12), (0.83, 0.74, 1.50)],
                0.10,
                0.14,
                OUT,
                "boron",
                [DOOR_FRAME, DOOR_FRAME],
                0,
                true,
                3,
            ))),
            DOOR_STATION,
        ),
        catching(stations(
            riding(ladder(member(
                "door_beam_r",
                &[(0.83, 0.42, 2.10), (0.83, 0.58, 1.52)],
                0.10,
                0.14,
                OUT,
                "boron",
                [DOOR_FRAME, DOOR_FRAME],
                0,
                true,
                3,
            ))),
            DOOR_STATION,
        )),
    ];
    let mut members = members;
    // The doors' inner panels' end faces (hinge face and lock face): each
    // joins its door's member ends, so the members' pull reaches the hinges
    // and latch as a real door's inner panel carries it. Its two chords
    // share the face's yield in tension: an HSLA inner panel (σ_y 340 MPa)
    // 0.8 mm thick over its 150 mm deep face, 41 kN (above the hinges' and
    // latch's: the hardware lets go first).
    let face = 0.5 * 340e6 * 0.0008 * 0.15 / Grades::default().hsla.stiffer(FRAME_STIFF).deform;
    // (Each from its door's intrusion beam's end up to its belt's.)
    for (name, lo, z) in [
        ("door_face_hf", 0.42, [3.18, 3.22, 3.20]),
        ("door_face_lf", 0.62, [2.40, 2.36, 2.36]),
        ("door_face_hr", 0.42, [2.10, 2.12, 2.12]),
        ("door_face_lr", 0.58, [1.52, 1.50, 1.48]),
    ] {
        members.push(stations(
            riding(ladder(member(
                name,
                &[(0.83, lo, z[0]), (0.83, 0.75, z[1]), (0.83, 0.93, z[2])],
                0.10,
                0.14,
                OUT,
                "hsla",
                [face, face],
                0,
                true,
                3,
            ))),
            DOOR_STATION,
        ));
    }
    // The boot lid's and bonnet's inner frames: a ladder across under each
    // row of the skin (its upper chord the skin's nodes, its lower 6 cm
    // below), the panel's grade in the skin's tier: the single sheet gets
    // depth, as the panel's inner stamping gives it.
    // Each chord yields at the frame's plastic moment over its lever: a hat
    // 100 wide × 60 deep of the 0.75 mm sheet at σ_y 210 MPa (a
    // bake-hardening closure grade), M_p = σ_y·Z_p = 1.2 kN·m, 20 kN a chord.
    let lid_sec = Section {
        depth: 0.06,
        width: 0.10,
        t: 0.00075,
        sigma_y: 210e6,
        round: false,
    };
    let lid_chord = lid_sec.m_p() / 0.06 / Grades::default().panel.deform;
    let inner = |name: &str, z: f32, x: f32, y: [f32; 2]| {
        let mut m = stations(
            riding(ladder(member(
                name,
                &[
                    (-x, y[0] - 0.03, z),
                    (0.0, y[1] - 0.03, z),
                    (x, y[0] - 0.03, z),
                ],
                0.10,
                0.06,
                UP,
                "panel",
                [lid_chord, lid_chord],
                0,
                false,
                3,
            ))),
            x,
        );
        m.skin = true;
        m
    };
    members.extend([
        inner("lid_inner_r", 0.14, 0.58, [1.00, 1.04]),
        inner("lid_inner_f", 0.46, 0.57, [1.03, 1.07]),
        inner("bonnet_inner_1", 3.79, 0.61, [0.94, 0.97]),
        inner("bonnet_inner_3", 4.30, 0.59, [0.85, 0.88]),
    ]);
    // The transverse members a sedan has besides those above (their
    // sections in [`side_sections`]): the front subframe's rear cross
    // member and the rear seat cushion's cross member. (A boot floor cross
    // member rail to rail at z 0.40 was tried: it moved the frontal's
    // restitution to 0.202 and did not stop the rear overhang's
    // pass-through; `vehicle-authoring.md`.)
    members.extend([
        ladder(member(
            "subframe_f0",
            &[(-0.44, 0.29, 3.55), (0.0, 0.27, 3.55), (0.44, 0.29, 3.55)],
            0.10,
            0.18,
            UP,
            "hsla",
            [1.0, 1.0],
            0,
            false,
            2,
        )),
        cell(floor_cross("seat_x", 1.58, "uhss")),
    ]);
    // The side's load path at its sections' strength.
    sectioned(&mut members, &Grades::default());
    let blocks = vec![
        BlockSpec {
            name: "powertrain".into(),
            centre: [0.0, 0.56, 3.98],
            half: [0.33, 0.24, 0.23],
            // Under a pole's width (static contact is per node).
            spacing: 0.17,
            // Two layers deep: its front face meets the wall and its rear
            // face the firewall, both at 0.17 m; nothing loads its middle.
            depth: 0.46,
            stiff: POWERTRAIN_STIFF,
            kg: 175.0,
            mounts: 2,
            reach: 0.25,
            mount_grade: "mount".into(),
            // Mounts that hold the engine in its bay (≈ 15 kN each, about a
            // real mount bracket's strength): an off-centre pole drives it
            // back into the firewall instead of sliding it aside (which let
            // the pole through between it and a rail).
            mount_scale: 4.0,
            bear: 0.5,
        },
        // Seat frames (the seats ride them): as heavy as their own lattice
        // needs; the rest of the seats' mass is a named mass on the floor
        // (`front_seats`, `rear_seat`), where it steadies the structure
        // instead of springing on its mounts in a crash.
        seat_frame("seat_fl", [0.38, 0.40, 2.62], [0.24, 0.07, 0.24], 7.5),
        seat_frame("seat_fr", [-0.38, 0.40, 2.62], [0.24, 0.07, 0.24], 7.5),
        seat_frame("seat_rear", [0.0, 0.40, 1.74], [0.62, 0.07, 0.22], 12.5),
        BlockSpec {
            name: "battery".into(),
            centre: [0.56, 0.62, 4.24],
            half: [0.09, 0.08, 0.11],
            spacing: 0.3,
            depth: 0.0,
            stiff: 1.0,
            kg: 18.0,
            mounts: 2,
            reach: 0.25,
            mount_grade: "mount".into(),
            mount_scale: 3.0,
            bear: 0.0,
        },
    ];
    let plates = vec![PlateSpec {
        name: "radiator".into(),
        corner: [-0.40, 0.34, 4.32],
        u: [0.80, 0.0, 0.0],
        v: [0.0, 0.34, 0.0],
        kg: 10.0,
        grade: "alu".into(),
        mount_scale: 0.4,
    }];
    // Each fills the stability floors of the frame nodes in its box first
    // (`Skeleton::add_mass_by_need`): the boxes span the structure that
    // carries it (seats and occupants: the floor members, sills, tunnel and
    // the pillars' feet, where the belts anchor; trim: the whole cabin).
    let masses = vec![
        NamedMass {
            name: "driver".into(),
            kg: 75.0,
            lo: [0.12, 0.12, 2.24],
            hi: [0.85, 1.0, 3.05],
        },
        NamedMass {
            name: "front_seats".into(),
            kg: 35.0,
            lo: [-0.85, 0.12, 2.24],
            hi: [0.85, 1.0, 3.05],
        },
        NamedMass {
            name: "rear_seat".into(),
            kg: 17.5,
            lo: [-0.85, 0.12, 1.30],
            hi: [0.85, 1.0, 2.05],
        },
        NamedMass {
            name: "dashboard".into(),
            kg: 30.0,
            lo: [-0.8, 0.7, 3.05],
            hi: [0.8, 0.95, 3.5],
        },
        NamedMass {
            name: "fuel".into(),
            kg: 45.0,
            lo: [-0.6, 0.12, 1.15],
            hi: [0.6, 0.36, 1.7],
        },
        NamedMass {
            name: "interior".into(),
            kg: 50.0,
            lo: [-0.9, 0.12, 1.30],
            hi: [0.9, 1.45, 3.40],
        },
        // Muffler and pipes (about 15 kg) along the tunnel and under the
        // rear floor.
        NamedMass {
            name: "exhaust".into(),
            kg: 15.0,
            lo: [-0.5, 0.12, 0.2],
            hi: [0.5, 0.40, 3.3],
        },
        NamedMass {
            name: "spare_wheel".into(),
            kg: 15.0,
            lo: [-0.5, 0.15, 0.3],
            hi: [0.5, 0.5, 0.8],
        },
    ];
    let wheel = |z: f32, driven: bool, spring: f32, damper: f32, cross: f32| Corner {
        x: 0.795,
        y: 0.335,
        z,
        tyre_radius: 0.335,
        tyre_width: 0.22,
        wheel_mass: 22.0,
        hub_mass: 14.0,
        arm_mass: 10.0,
        rays: 14,
        tyre_spring: 800_000.0,
        tyre_damp: 150.0,
        spring,
        damper,
        driven,
        braked: true,
        cross,
    };
    let look = Look {
        paint: [0.42, 0.015, 0.025],
        side_glass,
        windscreen: [2.55, 3.33],
        backlight: [0.78, 1.38],
        frit: 0.05,
        headlight: Outline::new(&[
            (0.44, 0.66),
            (0.80, 0.70),
            (0.86, 0.76),
            (0.56, 0.78),
            (0.40, 0.74),
        ]),
        grille: Outline::new(&[(0.0, 0.47), (0.36, 0.48), (0.40, 0.66), (0.0, 0.68)]),
        intake: Outline::new(&[(0.0, 0.26), (0.52, 0.27), (0.48, 0.37), (0.0, 0.37)]),
        taillight: Outline::new(&[(0.42, 0.83), (0.86, 0.84), (0.86, 0.96), (0.50, 0.97)]),
        handles: vec![[2.42, 0.92], [1.56, 0.93]],
        gap: 0.005,
        plate_y: [0.40, 0.70],
        mirror: [3.22, 1.0],
    };
    restyle(VehicleDef {
        name: "Rubble Showcase Sedan".into(),
        profile,
        grades: {
            let mut g = Grades::default();
            // The frame's grades at the rigidity its heavier nodes pay
            // for, the crush grade's at what its measured springback needs.
            for m in [&mut g.hsla, &mut g.uhss, &mut g.boron, &mut g.alu] {
                *m = m.stiffer(FRAME_STIFF);
            }
            g.crush = g
                .crush
                .scaled(CRUSH_PLATEAU)
                .stiffer(FRAME_STIFF * CRUSH_RIGIDITY);
            g
        },
        // Door seams (each door's edge ring inside it, the pillar's between),
        // screens' ends, the bonnet's and boot lid's edges.
        stations: vec![
            0.78, 1.30, 1.52, 2.06, 2.24, 2.42, 2.59, 2.74, 3.16, 3.32, 3.48, 4.10,
        ],
        panels,
        shell_grade: "panel".into(),
        zones,
        crumple: Curve::new(&[
            (0.00, 0.45),
            (0.45, 0.6),
            (0.90, 0.9),
            (1.30, 1.6),
            (1.45, 2.2),
            (3.30, 2.2),
            (3.45, 1.2),
            (3.90, 0.8),
            (4.40, 0.55),
            (4.70, 0.45),
        ]),
        members,
        blocks,
        plates,
        masses,
        corners: vec![
            wheel(FRONT_AXLE, true, 30_000.0, 2_200.0, 0.0),
            wheel(1.03, false, 26_000.0, 2_000.0, REAR_CROSS),
        ],
        kerb_mass: 1450.0,
        look,
        frame_every: 0,
        frame_grade: "uhss".into(),
        webs: vec![
            ("apron".into(), "rail_f".into(), 0.6),
            ("quarter_inner".into(), "rail_r".into(), 0.6),
            ("quarter_inner".into(), "rail_r_tail".into(), 0.6),
        ],
        // The cabin as a braced box: its shear panels (floor pan, roof,
        // firewall and screen, rear bulkhead, each door ring's frame) lumped
        // into X's that share the box's corners (sill and roof-rail ends at
        // the A-, B- and C-pillars), so the box closes: a torsion load meets
        // every face at once.
        braces: {
            let k = KL_PANEL;
            let (fb, ft, cw) = ([0.79, 0.31, 3.36], [0.62, 1.37, 2.59], [0.79, 0.92, 3.38]);
            let (bb, bt) = ([0.79, 0.31, 2.24], [0.62, 1.39, 2.24]);
            let (rb, rt) = ([0.79, 0.31, 1.36], [0.62, 1.37, 1.50]);
            // Front end: strut-tower tops and bases, the rails at the firewall.
            let (tf, tfb) = ([0.66, 0.82, FRONT_AXLE], [0.44, 0.41, FRONT_AXLE]);
            let rail_fw = [0.44, 0.31, 3.36];
            // Rear end: strut-tower tops and bases, the parcel shelf, the rear
            // rails.
            let (tr, trb) = ([0.68, 0.86, 1.03], [0.44, 0.40, 1.03]);
            let parcel = [0.79, 0.98, 1.36];
            let (rr1, rr2) = ([0.44, 0.24, 1.36], [0.44, 0.42, 0.62]);
            // The tunnel's ends.
            let (tun_r, tun_f) = ([0.0, 0.33, 1.36], [0.0, 0.33, 3.36]);
            let m = |p: [f32; 3]| [-p[0], p[1], p[2]];
            let _ = (bb, bt, ft, rt, tun_r, tun_f);
            cabin_x(fb, ft, cw, bb, bt, rb, rt, k)
                .into_iter()
                .chain([yielding(
                    brace("hinge_x", cw, bb, "uhss", k * HINGE_PANEL),
                    HINGE_YIELD,
                )])
                .chain({
                    [
                        // Front end: the cowl and strut towers as one box (strut
                        // brace, towers to the cowl and firewall, the bay's floor).
                        yielding(brace("strut_x", tf, m(cw), "uhss", k), END_YIELD),
                        brace("strut_bar", tf, m(tf), "uhss", k),
                        yielding(brace("strut_cowl", tf, cw, "uhss", k), END_YIELD),
                        yielding(brace("strut_fw", tf, fb, "uhss", k), END_YIELD),
                        yielding(brace("tower_cowl", tfb, cw, "uhss", k), END_YIELD),
                        yielding(brace("bay_floor", tfb, m(rail_fw), "uhss", k), END_YIELD),
                        // Rear end: towers, parcel shelf, C-pillars, rear floor.
                        brace("rtower_x", tr, m(parcel), "uhss", k),
                        brace("rtower_bar", tr, m(tr), "uhss", k),
                        brace("rtower_parcel", tr, parcel, "uhss", k),
                        brace("rtower_c", tr, rb, "uhss", k),
                        brace("rtower_base", trb, parcel, "uhss", k),
                        brace("rfloor_x", rr1, m(rr2), "floor", k),
                        brace("rbase_x", trb, m(rr1), "floor", k),
                    ]
                })
                // Each end on the member its corner lies on (explicit ends:
                // a member added elsewhere never moves a brace).
                .map(|b| {
                    let hosts = [
                        (fb, "pillar_a"),
                        (ft, "roof_rail"),
                        (cw, "apron"),
                        (bb, "pillar_b"),
                        (bt, "roof_rail"),
                        (rb, "pillar_c"),
                        (rt, "pillar_c"),
                        (tf, "apron"),
                        (tfb, "rail_f"),
                        (rail_fw, "rail_f"),
                        (tr, "tower_r"),
                        (trb, "tower_r"),
                        (parcel, "pillar_c"),
                        (rr1, "kickup"),
                        (rr2, "rail_r"),
                    ];
                    let host = |p: [f32; 3]| {
                        hosts
                            .iter()
                            .find(|(q, _)| q[0] == p[0].abs() && q[1] == p[1] && q[2] == p[2])
                            .map(|x| x.1)
                            .expect("a brace corner on a member")
                    };
                    if !b.on[0].is_empty() {
                        return b;
                    }
                    let (a, c) = (host(b.from), host(b.to));
                    on(b, a, c)
                })
                .collect()
        },
        shear: {
            let k = KL_SHEAR;
            let sh = |a: &str, b: &str, axis: usize, grade: &str| ShearSpec {
                a: a.into(),
                b: b.into(),
                axis,
                grade: grade.into(),
                kl: k,
                // Screens crack.
                strength: if grade == "glass" { GLASS_YIELD } else { 0.0 },
            };
            vec![
                // The cabin box's faces.
                sh("roof_rail", "roof_rail_r", 2, "panel"),
                sh("sill", "tunnel", 2, "floor"),
                sh("sill", "roof_rail", 2, "uhss"),
                sh("firewall", "cowl", 0, "hsla"),
                sh("cowl", "header_f", 0, "glass"),
                sh("kickup", "parcel", 0, "hsla"),
                sh("parcel", "header_r", 0, "glass"),
                // Front end: inner fenders, the bay's floor.
                sh("rail_f", "apron", 2, "hsla"),
                sh("subframe_f1", "subframe_f2", 0, "hsla"),
                // Rear end: rear floor, inner quarters, subframe.
                sh("rail_r", "rail_r_r", 2, "floor"),
                sh("rail_r_tail", "rail_r_tail_r", 2, "floor"),
                // The rear quarter (wheelhouse and quarter inner) between
                // the rear rail and the quarter inner: the rear body's deep
                // beam's web.
                {
                    let mut x = sh("rail_r", "quarter_inner", 2, "hsla");
                    x.kl *= QUARTER_WEB;
                    x
                },
                {
                    let mut x = sh("rail_r_tail", "quarter_inner", 2, "hsla");
                    x.kl *= QUARTER_WEB;
                    x
                },
                // The floor pan in shear between the rails and the sills:
                // the rails' crash load into the sills.
                sh("rail_f", "sill", 2, "floor"),
                sh("rail_r", "sill", 2, "floor"),
                sh("subframe_r1", "subframe_r2", 0, "hsla"),
            ]
        },
        frame_through: vec!["door".into()],
        // The bonnet too (its inner panel bears on the aprons and towers
        // under its edges): crushed, it folds over the engine bay instead of
        // sinking through the wheelhouses' skin (which then rattles in
        // self-contact).
        backed: vec![
            "door".into(),
            "fender".into(),
            "bumper".into(),
            "bonnet".into(),
            // The boot lid's edges rest on the deck opening's gutter.
            "bootlid".into(),
        ],
        shell_braces: false,
        body_damping: 50.0,
        tiered: true,
        body_damping_reach: 2.0,
        body_damping_strength: 5000.0,
        flanges: door_flanges(),
        sheets: panel_sheets(),
        one_tier: false,
        biw_mass: 0.0,
        part_masses: Vec::new(),
        damping: None,
        crush: Vec::new(),
    })
}

/// The 2011 Honda Accord's painted body-in-white with its sprayed-on sound
/// deadening (kg): NHTSA DOT HS 811 666 §5.9 (328 kg structure + 12 kg).
pub const ACCORD_BIW: f32 = 339.0;
/// The Accord's closures and fenders as assemblies (kg each: structure,
/// glass, regulators, seals, wiring, speakers, hinges and latch, mirrors,
/// trim and fasteners), DOT HS 811 666 §5.10: front doors 58.99 the pair
/// (fig. 161), rear doors 47.46 the pair (fig. 171), hood 17.90, decklid
/// 12.37; the fenders the rest of the closures' 147 kg, a pair.
pub const ACCORD_DOOR_F: f32 = 58.99 / 2.0;
pub const ACCORD_DOOR_R: f32 = 47.46 / 2.0;
pub const ACCORD_HOOD: f32 = 17.90;
pub const ACCORD_DECKLID: f32 = 12.37;
pub const ACCORD_FENDER: f32 = (147.0 - 58.99 - 47.46 - 17.90 - 12.37) / 2.0;

/// A section's steel per metre (kg/m): a box's perimeter × wall, a tube's
/// circumference × wall, × 7,850 kg/m³.
pub fn section_kg_per_m(s: &Section) -> f32 {
    let perimeter = if s.round {
        std::f32::consts::PI * s.depth
    } else {
        2.0 * (s.depth + s.width)
    };
    perimeter * s.t * super::members::STEEL_DENSITY
}

/// The showcase sedan built in one tier, as BeamNG's cars are
/// ([`VehicleDef::one_tier`]): the body's shell is its frame, every node
/// weighs its real local mass, and the beams are as stiff as their materials
/// unless the step's bound requires less. The same shape, members, parts and
/// hardware as [`sedan`]; the body-in-white's and the closures' masses are
/// the 2011 Accord teardown's (NHTSA DOT HS 811 666).
pub fn sedan_one_tier() -> VehicleDef {
    let mut d = sedan();
    d.name = "sedan1".into();
    d.one_tier = true;
    d.tiered = false;
    d.damping = Some(super::vehicle::DampingRatios::default());
    // The front bumper beam: the 2011 Accord's is roll-formed steel of
    // 590 MPa tensile strength, its assembly with crush cans, centre plate
    // and brackets 7.96 kg (NHTSA DOT HS 811 666 §5.10.7.1). Its section: a
    // closed 60 deep × 100 tall roll-formed box, 1.6 mm (a Choice: about
    // 4.0 kg/m, so some 5 kg of the assembly over its 1.2 m), at a 590
    // grade's yield (DP600's 330 MPa). Its bending yields from the
    // section's plastic moment, as the rear's.
    {
        let beam = Section {
            depth: 0.06,
            width: 0.10,
            t: 0.0016,
            sigma_y: DP600,
            round: false,
        };
        sectioned_by(
            &mut d.members,
            &Grades::default(),
            &[("bumper_beam_f", beam, 0.12, 1.0)],
        );
        for m in d.members.iter_mut().filter(|m| m.name == "bumper_beam_f") {
            m.kg_per_m = section_kg_per_m(&beam);
        }
    }
    // Crush members: the front rails (an 80 × 120 × 1.6 mm DP600 box, a
    // Choice of a typical rail's section; σ_u 600 MPa, EN 10338) and the
    // rear rails' tail behind the rear subframe ([`side_sections`]' rear
    // rail, 100 × 130 × 1.6 mm DP600) fold at Abramowicz and Jones's mean
    // force (61.8 and 64.7 kN); the bumper beams keep their bending
    // yields and fold without hardening. All stroke to
    // [`super::vehicle::CRUSH_STROKE`].
    let pm = |b: f32, dd: f32, t: f32| {
        super::vehicle::progressive_crush_force(b, dd, t, DP600, DP600_UTS)
    };
    let crush = |member: &str, mean_force: f32| super::vehicle::CrushSpec {
        member: member.into(),
        mean_force,
        deform_limit: super::vehicle::CRUSH_STROKE,
    };
    d.crush = vec![
        crush("rail_f", pm(0.08, 0.12, 0.0016)),
        crush("rail_r_tail", pm(0.10, 0.13, 0.0016)),
        crush("bumper_beam_f", 0.0),
        crush("bumper_beam_r", 0.0),
    ];
    for m in d.members.iter_mut().filter(|m| m.name == "rail_f") {
        // (The bays' cuts set the strength: no tuned grading.)
        m.scale = [1.0, 1.0];
    }
    d.biw_mass = ACCORD_BIW;
    d.part_masses = vec![
        ("door_f".into(), ACCORD_DOOR_F),
        ("door_r".into(), ACCORD_DOOR_R),
        ("bonnet".into(), ACCORD_HOOD),
        ("bootlid".into(), ACCORD_DECKLID),
        ("fender".into(), ACCORD_FENDER),
    ];
    for (name, sec, _, _) in side_sections() {
        for m in d.members.iter_mut().filter(|m| m.name == name) {
            m.kg_per_m = section_kg_per_m(&sec);
        }
    }
    d
}

/// The bolted and hinged panels' sheets: steel outers 0.75 mm (outer panel
/// gauge 0.7–0.9 mm, `beamng-realism-plan.md` [GAUGE]); the bonnet's and
/// boot lid's inner frames a stamping of the same gauge over half the area
/// (its cut-outs the rest); bumper fascias 3 mm of polypropylene compound
/// (1,000 kg/m³). Steel 7,850 kg/m³.
pub fn panel_sheets() -> Vec<SheetSpec> {
    let steel = 7850.0;
    let outer = [0.00075, 1.0, steel];
    let inner = [0.00075, 0.5, steel];
    let fascia = [0.003, 1.0, 1000.0];
    let s = |part: &str, layers: &[[f32; 3]]| SheetSpec {
        part: part.into(),
        layers: layers.to_vec(),
    };
    vec![
        s("bonnet", &[outer, inner]),
        s("bootlid", &[outer, inner]),
        s("fender", &[outer]),
        s("bumper", &[fascia]),
    ]
}

/// The pillars' door flanges a door member's end bears on (each over
/// 50 mm, the end's bracket): the hinge pillar's DP980 and the B-pillar's
/// hot-stamped reinforcement at 1.8 mm (the gauges and grades of
/// [`side_sections`]' sill and B-pillar), the C-pillar's DP780 at 1.4 mm.
/// Capacities σ_y·t·w: 53, 99 and 31 kN.
pub fn door_flanges() -> Vec<FlangeSpec> {
    let (dp980, phs, dp780) = (590e6, 1100e6, 440e6);
    let f = |member: &str, t: f32, sigma_y: f32| FlangeSpec {
        member: member.into(),
        t,
        width: 0.05,
        sigma_y,
    };
    vec![
        f("pillar_a", 0.0018, dp980),
        f("pillar_b", 0.0018, phs),
        f("pillar_c", 0.0014, dp780),
    ]
}
