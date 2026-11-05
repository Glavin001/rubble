//! The sedan's styling and the layout of its parts.
//!
//! [`Styling`] holds every visual design number (shut lines, openings, lamp
//! outlines, trim) in the definition's coordinates; [`assemble`] builds the
//! body surface, cuts the panels, and places the fixtures. A new body style
//! changes these numbers (and its `BodyProfile`), not the tools.
//!
//! Views used by the outlines:
//! * side view (z, y);
//! * plan (x, z);
//! * *arc* view (a, h): a = arc length round the plan outline from the front
//!   centre (or from the rear centre for the tail), h = height relative to the
//!   band's top edge at that place. Lamps and grilles are drawn here, so they
//!   wrap round the corners exactly as the body does.

use super::body::{BodyStyle, BodySurface, SectionShape};
use super::trim::{self, EdgeStyle};
use super::{fixtures, smax, smin, Bind, Mesh, Piece, Shape};
use crate::authoring::curve::Curve;
use crate::authoring::skin::mat;
use crate::authoring::vehicle::VehicleDef;
use glam::Vec3;

/// The visual wheels sit this far (m) outboard of the physics tyres.
pub const WHEEL_OUT: f32 = 0.042;

/// Visual design inputs.
#[derive(Clone, Debug)]
pub struct Styling {
    pub body: BodyStyle,
    /// Bonnet: rear (cowl) edge z, lateral cut beyond the top line, front edge
    /// above the band's top.
    pub cowl: [f32; 2],
    pub hood_side: f32,
    /// Door cuts (side view, z at the sill line and the slope dz/dy): front door
    /// front edge, B-pillar, rear door rear edge; sill cut height.
    pub door_front: [f32; 2],
    pub b_cut: [f32; 2],
    pub door_rear: [f32; 2],
    pub sill: f32,
    /// Front bumper / fender cut: z at y = 0.52 and dz/dy.
    pub bumper_f: [f32; 2],
    /// Rear bumper top (y) and its side cut (z at that height, dz/dy).
    pub bumper_r: [f32; 3],
    /// Deck lid: front edge z, half width on top and on the tail face, lower edge y.
    pub deck: [f32; 4],
    /// Glass: windscreen (z0, z1), backlight (z0, z1), side windows' z spans
    /// (front, rear, quarter), A-pillar inset on top, rail inset.
    pub windscreen: [f32; 2],
    pub backlight: [f32; 2],
    pub side_windows: [[f32; 2]; 3],
    pub pillar: f32,
    pub rail: f32,
    pub frame: f32,
    /// Arc-view outlines (a, h): headlight, grille, intake, fog; tail light (a
    /// from the rear centre).
    pub headlight: Vec<(f32, f32)>,
    pub grille: Vec<(f32, f32)>,
    pub intake: Vec<(f32, f32)>,
    pub taillight: Vec<(f32, f32)>,
    /// The front plate's centre height (relative to the band's top); NaN for
    /// none.
    pub plate_front_h: f32,
    /// Fog lamps (arc view, from the front centre).
    pub fog: Vec<(f32, f32)>,
    /// The boot lid's light bar: half length (arc from the rear centre), centre
    /// height and half height (relative to the band's top).
    pub lightbar: [f32; 3],
    pub arch_radius: f32,
    /// Exhaust tips: |x|, y, radius (rear view).
    pub exhaust: [f32; 3],
    pub paint: [f32; 3],
}

pub fn sedan_styling(def: &VehicleDef) -> Styling {
    let side = SectionShape {
        lower: 0.11,
        d_lower: -0.065,
        crease: f32::NAN,
        a_below: 0.04,
        a_above: 0.85,
        d_top: -0.06,
        m_top: 0.25,
        tuck: 0.07,
        bow: 2.4,
    };
    // An upright fascia: its widest line high (at the lamps' lower edge), the
    // lower bumper drawn back under it (a shorter-looking overhang), and a
    // crisp turn into the bonnet.
    let nose = SectionShape {
        lower: 0.05,
        d_lower: -0.04,
        crease: 0.56,
        a_below: 0.07,
        a_above: 0.45,
        d_top: -0.14,
        m_top: 0.7,
        tuck: 0.04,
        bow: 0.0,
    };
    let tail = SectionShape {
        lower: 0.08,
        d_lower: -0.05,
        crease: 0.70,
        a_below: 0.26,
        a_above: 0.34,
        d_top: -0.03,
        m_top: 0.2,
        tuck: 0.07,
        bow: 0.0,
    };
    let body = BodyStyle {
        // The shoulder: one line from above the headlamp, level through the
        // handles, kicking up into the tail lamps.
        crease: Curve::new(&[
            (0.00, 0.86),
            (0.30, 0.90),
            (0.75, 0.905),
            (1.20, 0.88),
            (1.60, 0.835),
            (2.20, 0.815),
            (2.90, 0.805),
            (3.50, 0.79),
            (4.10, 0.75),
            (4.70, 0.68),
        ]),
        side,
        nose,
        tail,
        nose_bow: 0.11,
        tail_bow: 0.09,
        nose_corner: 0.47,
        tail_corner: 0.31,
        flare: 0.018,
        flare_width: 0.19,
        rear_flare: 0.036,
        lid_crown: 0.034,
        step: 0.022,
        step_along: 0.035,
        nose_ramp: 0.045,
        tail_ramp: 0.04,
        rear_deck: [0.85, 1.50, 0.045, 0.035],
        hood_crease: [0.22, 0.60, 0.014, 0.004],
        hood_z: [3.36, 4.70],
        deck_lip: [0.026, 0.24, 0.60],
        // Straight, rising 3.5 cm to the rear: parallel to the rocker's line
        // in side view, not diving toward the front arch.
        scoop: Curve::new(&[(1.30, 0.495), (3.45, 0.46)]),
        scoop_depth: 0.024,
        scoop_z: [1.42, 1.80, 3.10, 3.40],
    };
    let _ = def;
    Styling {
        body,
        cowl: [3.32, 3.41],
        hood_side: 0.035,
        door_front: [3.30, -0.08],
        b_cut: [2.24, 0.02],
        door_rear: [1.46, 0.10],
        sill: 0.265,
        bumper_f: [4.10, 1.15],
        bumper_r: [0.70, 0.40, 1.25],
        deck: [0.79, 0.62, 0.47, 0.71],
        windscreen: [2.59, 3.32],
        backlight: [0.84, 1.36],
        // The B-pillar blackout 8 cm (2.20..2.28); the rear door's glass and
        // the quarter light split by one slim sash (2 cm).
        side_windows: [[2.28, 3.25], [1.62, 2.20], [1.12, 1.60]],
        pillar: 0.035,
        rail: 0.045,
        frame: 0.010,
        // Slim, sharp full-LED blades just under the bonnet's leading edge,
        // rising outward and wrapping round the corner to a pointed tip.
        headlight: vec![
            (0.40, -0.056),
            (0.40, -0.020),
            (0.62, -0.013),
            (0.88, -0.009),
            (1.06, -0.008),
            (1.125, -0.012),
            (1.115, -0.019),
            (1.04, -0.024),
            (0.86, -0.033),
            (0.62, -0.046),
        ],
        // A wide, low grille between the lamps' inner ends and the bumper.
        // A trapezoid, narrow at the top and widening toward the bumper's
        // lower edge, so the face's dark mass sits low.
        // Its top edge just under the lamps' inner ends (no blank band of
        // paint between lamp and grille).
        grille: vec![
            (-0.33, -0.084),
            (0.33, -0.084),
            (0.372, -0.090),
            (0.392, -0.112),
            (0.385, -0.13),
            (0.43, -0.235),
            (0.445, -0.272),
            (0.425, -0.292),
            (0.38, -0.300),
            (-0.38, -0.300),
            (-0.425, -0.292),
            (-0.445, -0.272),
            (-0.43, -0.235),
            (-0.385, -0.13),
            (-0.392, -0.112),
            (-0.372, -0.090),
        ],
        // Side intakes in the bumper's corners (air curtains): sharp
        // trapezoids, the inner edge raked.
        // A blade in the bumper's corner: a slim slot along the lower edge
        // rising into a tall air curtain at the outer end (an L in arc view),
        // so the bumper's corner stays a sculpted surface, not a hole.
        intake: vec![
            (0.60, -0.318),
            (0.63, -0.300),
            (0.80, -0.286),
            (0.835, -0.270),
            (0.85, -0.18),
            (0.862, -0.160),
            (0.882, -0.168),
            (0.900, -0.250),
            (0.892, -0.322),
            (0.872, -0.340),
            (0.66, -0.342),
            (0.612, -0.334),
        ],
        // Wrapped LED tail lamps: deep at the outer end, a sharp tip round
        // the corner, joined by the light bar across the boot lid.
        taillight: vec![
            (0.465, -0.012),
            (0.75, -0.008),
            (1.02, -0.012),
            (1.20, -0.026),
            (1.255, -0.045),
            (1.24, -0.075),
            (1.10, -0.112),
            (0.98, -0.118),
            (0.72, -0.100),
            (0.465, -0.092),
        ],
        lightbar: [0.445, -0.051, 0.036],
        // No front plate: the grille is the face.
        plate_front_h: f32::NAN,
        fog: Vec::new(),
        arch_radius: 0.338,
        exhaust: [0.50, 0.44, 0.046],
        paint: def.look.paint,
    }
}

/// A body vertex's views.
struct At<'a> {
    s: &'a BodySurface,
}

impl At<'_> {
    fn ytop(&self, i: usize) -> f32 {
        self.s.ytop[i]
    }
}

/// Build every visual piece of the car.
pub fn assemble(def: &VehicleDef, st: &Styling) -> Vec<Piece> {
    let prof = &def.profile;
    // The distant LOD samples the body coarser (cut lines stay exact).
    let mut body_style = st.body.clone();
    body_style.step /= super::detail().max(0.25);
    body_style.step_along /= super::detail().max(0.25);
    let surf = BodySurface::build(prof, &body_style);
    let at = At { s: &surf };
    let m = &surf.mesh;
    let half = surf.band.half;
    let mut pieces = Vec::new();
    // --- Region terms ------------------------------------------------------
    let lidrow = |z: f32| surf.lid_row(z);
    let arch = |p: Vec3| -> f32 {
        if p.x.abs() < 0.45 {
            return 1.0;
        }
        prof.wells
            .iter()
            .map(|w| glam::Vec2::new(p.z - w.z, p.y - w.y).length() - st.arch_radius)
            .fold(f32::MAX, f32::min)
    };
    let mirror_a = |a: &[(f32, f32)]| -> Vec<(f32, f32)> { a.to_vec() };
    let head = Shape::smooth(&mirror_a(&st.headlight));
    let grille = Shape::smooth(&st.grille);
    let intake = Shape::smooth(&st.intake);
    let tail = Shape::smooth(&st.taillight);
    // Arc-view coordinates: (a from the front, h) and (a from the rear, h).
    let front_ah = |i: usize, p: Vec3| -> (f32, f32) { (surf.arc[i], p.y - at.ytop(i)) };
    let rear_ah = |i: usize, p: Vec3| -> (f32, f32) { (half - surf.arc[i], p.y - at.ytop(i)) };
    let on_band = |i: usize| !surf.lid[i];
    let headlamp = |i: usize, p: Vec3| -> f32 {
        if !on_band(i) {
            return 1.0;
        }
        let (a, h) = front_ah(i, p);
        head.sd(a, h)
    };
    let grille_f = |i: usize, p: Vec3| -> f32 {
        if !on_band(i) || surf.arc[i] > 0.7 {
            return 1.0;
        }
        let (a, h) = front_ah(i, p);
        grille.sd(a, h)
    };
    let intake_f = |i: usize, p: Vec3| -> f32 {
        if !on_band(i) || surf.arc[i] > 1.1 {
            return 1.0;
        }
        let (a, h) = front_ah(i, p);
        intake.sd(a, h)
    };
    let taillamp = |i: usize, p: Vec3| -> f32 {
        if !on_band(i) {
            return 1.0;
        }
        let (a, h) = rear_ah(i, p);
        tail.sd(a, h)
    };
    let fog = Shape::smooth(if st.fog.is_empty() {
        &[(9.0, 9.0), (9.1, 9.0), (9.0, 9.1)]
    } else {
        &st.fog
    });
    let foglamp = |i: usize, p: Vec3| -> f32 {
        if st.fog.is_empty() || !on_band(i) || surf.arc[i] > 1.0 {
            return 1.0;
        }
        let (a, h) = front_ah(i, p);
        fog.sd(a, h)
    };
    // A full-width light bar across the boot lid between the tail lamps.
    let lightbar = |i: usize, p: Vec3| -> f32 {
        if !on_band(i) || p.z > 0.6 {
            return 1.0;
        }
        let (a, h) = rear_ah(i, p);
        smax(
            a - st.lightbar[0],
            (h - st.lightbar[1]).abs() - st.lightbar[2],
            0.004,
        )
    };
    // Glass.
    let windscreen = |i: usize, p: Vec3| -> f32 {
        if !surf.lid[i] {
            return 1.0;
        }
        let r = lidrow(p.z);
        let f = smax(st.windscreen[0] - p.z, p.z - st.windscreen[1], 0.08);
        smax(f, p.x.abs() - (r.x_r - st.pillar), 0.10).max(r.y_t + 0.03 - p.y)
    };
    let backlight = |i: usize, p: Vec3| -> f32 {
        if !surf.lid[i] {
            return 1.0;
        }
        let r = lidrow(p.z);
        let f = smax(st.backlight[0] - p.z, p.z - st.backlight[1], 0.08);
        smax(f, p.x.abs() - (r.x_r - st.pillar - 0.01), 0.10).max(r.y_t + 0.03 - p.y)
    };
    let side_window = |k: usize, i: usize, p: Vec3| -> f32 {
        if !surf.lid[i] || p.x.abs() < 0.3 {
            return 1.0;
        }
        let r = lidrow(p.z);
        let [z0, z1] = st.side_windows[k];
        // The quarter light sits on the belt's kick up the deck shoulder.
        let yb = if k == 2 { surf.belt(p.z) } else { r.y_t };
        let fy = smax(yb + 0.018 - p.y, p.y - (r.y_e - st.rail), 0.05);
        let fz = if k == 2 {
            // Quarter glass: its rear edge sweeps back along the C-pillar's
            // slope to a point on the belt (the DLO's tail), its top edge
            // under the falling roof.
            let zc = z0 + 1.42 * (p.y - yb - 0.012);
            smax(zc - p.z, p.z - z1, 0.03)
        } else {
            smax(z0 - p.z, p.z - z1, 0.04)
        };
        smax(fy, fz, 0.05)
    };
    let glass_side = |i: usize, p: Vec3| -> f32 {
        side_window(0, i, p)
            .min(side_window(1, i, p))
            .min(side_window(2, i, p))
    };
    let b_pillar = |i: usize, p: Vec3| -> f32 {
        if !surf.lid[i] || p.x.abs() < 0.3 {
            return 1.0;
        }
        let r = lidrow(p.z);
        let z0 = st.side_windows[1][1];
        let z1 = st.side_windows[0][0];
        // Square to the belt (no rounding at the bottom: it reads as a
        // point once the doors move).
        let fy = smax(r.y_t + 0.018 - p.y, p.y - (r.y_e - st.rail), 0.004);
        smax(fy, smax(z0 - p.z, p.z - z1, 0.004), 0.004)
    };
    // Exhaust cut-outs in the rear valance (rear view circles).
    let exhaust_hole = |p: Vec3| -> f32 {
        if p.z > 0.35 {
            return 1.0;
        }
        glam::Vec2::new(p.x.abs() - st.exhaust[0], p.y - st.exhaust[1]).length()
            - st.exhaust[2]
            - 0.008
    };
    // Openings in the body panels (negative inside the hole).
    let openings = |i: usize, p: Vec3| -> f32 {
        let mut o = arch(p);
        o = o.min(headlamp(i, p) - 0.0);
        o = o.min(grille_f(i, p));
        o = o.min(intake_f(i, p));
        o = o.min(taillamp(i, p));
        o = o.min(lightbar(i, p));
        o = o.min(foglamp(i, p));
        o = o.min(windscreen(i, p));
        o = o.min(backlight(i, p));
        o = o.min(glass_side(i, p) - st.frame);
        o = o.min(b_pillar(i, p));
        o = o.min(exhaust_hole(p));
        o
    };
    let open = trim::field(m, openings);
    // --- Panels --------------------------------------------------------------
    let ytl = |p: Vec3| lidrow(p.z).y_t;
    let door_front_z = |y: f32| st.door_front[0] + st.door_front[1] * (y - 0.3);
    let b_z = |y: f32| st.b_cut[0] + st.b_cut[1] * (y - 0.3);
    let door_rear_z = |y: f32| st.door_rear[0] + st.door_rear[1] * (y - 0.6);
    let bumper_fz = |y: f32| st.bumper_f[0] + st.bumper_f[1] * (y - 0.52);
    let bumper_rz = |y: f32| st.bumper_r[1] + st.bumper_r[2] * (st.bumper_r[0] - y);
    type Region<'a> = Box<dyn Fn(usize, Vec3) -> f32 + 'a>;
    let hood: Region = Box::new(|_, p| {
        let r = lidrow(p.z);
        let f = smax(st.cowl[1] - p.z, p.x.abs() - (r.x_r + st.hood_side), 0.06);
        smax(f, r.y_t + 0.012 - p.y, 0.03)
    });
    let cowl: Region = Box::new(|i, p| {
        if !surf.lid[i] {
            return 1.0;
        }
        let r = lidrow(p.z);
        let f = smax(st.cowl[0] - p.z, p.z - st.cowl[1], 0.0);
        smax(f, p.x.abs() - (r.x_r + st.hood_side), 0.03)
    });
    let bumper_f: Region = Box::new(|_, p| bumper_fz(p.y) - p.z);
    let a_base = move |p: Vec3| -> f32 {
        // The A-pillar base (lid behind the cowl line): not fender.
        smax(ytl(p) + 0.012 - p.y, p.z - st.cowl[0] - 0.02, 0.02)
    };
    let fender: Region = Box::new(move |_, p| smax(door_front_z(p.y) - p.z, -a_base(p), 0.02));
    let in_door_band = move |p: Vec3| smax(st.sill - p.y, p.y - (ytl(p) + 0.004), 0.02);
    let door_f: Region = Box::new(move |_, p| {
        smax(
            smax(p.z - door_front_z(p.y), b_z(p.y) - p.z, 0.025),
            in_door_band(p),
            0.025,
        )
    });
    let rear_arch = prof
        .wells
        .iter()
        .min_by(|a, b| a.z.total_cmp(&b.z))
        .copied();
    let door_r: Region = Box::new(move |_, p| {
        let mut f = smax(
            smax(p.z - b_z(p.y), door_rear_z(p.y) - p.z, 0.025),
            in_door_band(p),
            0.025,
        );
        if let Some(w) = rear_arch {
            let r = glam::Vec2::new(p.z - w.z, p.y - w.y).length();
            f = smax(f, st.arch_radius + 0.07 - r, 0.04);
        }
        f
    });
    let deck: Region = Box::new(|_, p| {
        let xd = st.deck[2] + (st.deck[1] - st.deck[2]) * super::smoothstep(0.86, 0.95, p.y);
        smax(
            smax(p.z - st.deck[0], p.x.abs() - xd, 0.05),
            st.deck[3] - p.y,
            0.04,
        )
    });
    let bumper_r: Region = Box::new(|_, p| smax(p.z - bumper_rz(p.y), p.y - st.bumper_r[0], 0.04));
    // Black lower lip under the front bumper, black diffuser under the rear.
    let lip_f: Region = Box::new(|_, p| smax(bumper_fz(p.y) - p.z, p.y - 0.285, 0.02));
    // The black diffuser insert, rising round the exhaust tips into pockets.
    let valance_r: Region = Box::new(|_, p| {
        let x = p.x.abs();
        let pocket = 0.075
            * super::smoothstep(st.exhaust[0] - 0.16, st.exhaust[0] - 0.09, x)
            * (1.0 - super::smoothstep(st.exhaust[0] + 0.09, st.exhaust[0] + 0.16, x));
        smax(p.z - bumper_rz(p.y), p.y - 0.505 - pocket, 0.03)
    });
    let rocker: Region = Box::new(|_, p| {
        if p.x.abs() < 0.4 {
            return 1.0;
        }
        p.y - st.sill
    });
    // (name, region, material, sided)
    let panels: Vec<(&str, Region, u8, bool)> = vec![
        ("bonnet", hood, mat::PAINT, false),
        ("cowl", cowl, mat::TRIM, false),
        ("lip_f", lip_f, mat::TRIM, false),
        ("bumper_f", bumper_f, mat::PAINT, false),
        ("fender", fender, mat::PAINT, true),
        ("door_f", door_f, mat::PAINT, true),
        ("door_r", door_r, mat::PAINT, true),
        ("deck", deck, mat::PAINT, false),
        ("valance_r", valance_r, mat::TRIM, false),
        ("bumper_r", bumper_r, mat::PAINT, false),
        ("rocker", rocker, mat::TRIM, true),
        ("body", Box::new(|_, _| -1.0), mat::PAINT, false),
    ];
    let mut taken: Vec<f32> = vec![f32::MAX; m.len()];
    // Window openings: a lid point near a glass edge (their walls are black
    // rubber seals).
    let near_glass = |p: Vec3| -> bool {
        let r = lidrow(p.z);
        // From just under the belt (the doors' top edges meet the glass's
        // weatherstrip there).
        if p.y < r.y_t - 0.006 {
            return false;
        }
        let i = usize::MAX;
        let _ = i;
        let ws = smax(
            smax(st.windscreen[0] - p.z, p.z - st.windscreen[1], 0.08),
            p.x.abs() - (r.x_r - st.pillar),
            0.10,
        );
        let bl = smax(
            smax(st.backlight[0] - p.z, p.z - st.backlight[1], 0.08),
            p.x.abs() - (r.x_r - st.pillar - 0.01),
            0.10,
        );
        let fy = smax(r.y_t + 0.018 - p.y, p.y - (r.y_e - st.rail), 0.05);
        let side = st
            .side_windows
            .iter()
            .map(|w| smax(fy, smax(w[0] - p.z - 0.06, p.z - w[1] - 0.06, 0.04), 0.05))
            .fold(f32::MAX, f32::min);
        ws.min(bl).min(side - st.frame) < 0.04
    };
    let edge_style = |p: Vec3| -> EdgeStyle {
        if arch(p) < 0.012 {
            EdgeStyle::ARCH
        } else if near_glass(p) {
            EdgeStyle {
                seal: true,
                ..EdgeStyle::SHUT
            }
        } else {
            EdgeStyle::SHUT
        }
    };
    for (name, region, material, sided) in &panels {
        let raw = trim::field(m, |i, p| region(i, p));
        // Inside this region, outside every earlier panel and every opening.
        let f: Vec<f32> = (0..m.len())
            .map(|i| raw[i].max(-taken[i]).max(-open[i]))
            .collect();
        let f = metric(m, &f);
        for (k, v) in raw.iter().enumerate() {
            taken[k] = taken[k].min(*v);
        }
        let sides: &[f32] = if *sided { &[1.0, -1.0] } else { &[0.0] };
        for &sx in sides {
            let fs: Vec<f32> = if sx == 0.0 {
                f.clone()
            } else {
                f.iter()
                    .zip(&m.pos)
                    .map(|(v, p)| v.max(-p.x * sx))
                    .collect()
            };
            let level = -(EdgeStyle::SHUT.gap + EdgeStyle::SHUT.roll);
            let mut c = trim::clip(m, &fs, level);
            if c.mesh.is_empty() {
                continue;
            }
            c.mesh.mat.iter_mut().for_each(|x| *x = *material);
            // A soft shadow either side of the shut line, wide enough to read from
            // a few metres (the 4.5 mm gap alone vanishes at the 3/4 distance).
            trim::edge_ao(&mut c, level, 0.38, 0.013);
            let loops = std::mem::take(&mut c.loops);
            for (l, closed) in &loops {
                // Per vertex: arch lips get the big roll, the rest shut lines.
                let lp: Vec<u32> = l.clone();
                let mid = c.mesh.pos[lp[lp.len() / 2] as usize];
                let sty = edge_style(mid);
                roll_mixed(&mut c.mesh, &lp, *closed, &edge_style, sty);
            }
            let nm = if sx > 0.0 {
                format!("{name}_l")
            } else if sx < 0.0 {
                format!("{name}_r")
            } else {
                name.to_string()
            };
            pieces.push(Piece {
                name: nm,
                mesh: c.mesh,
                bind: Bind::Vote,
            });
        }
    }
    // --- Glass and frames -----------------------------------------------------
    let glass_regions: Vec<(&str, Box<dyn Fn(usize, Vec3) -> f32 + '_>, f32)> = vec![
        ("windscreen", Box::new(windscreen), 0.06),
        ("backlight", Box::new(backlight), 0.05),
        ("glass_side", Box::new(glass_side), 0.0),
    ];
    for (name, region, frit) in glass_regions {
        let f = metric(m, &trim::field(m, |i, p| region(i, p)));
        for sx in [1.0f32, -1.0] {
            let fs: Vec<f32> = if name == "glass_side" {
                f.iter()
                    .zip(&m.pos)
                    .map(|(v, p)| v.max(-p.x * sx))
                    .collect()
            } else if sx > 0.0 {
                f.clone()
            } else {
                continue;
            };
            let c = trim::clip(m, &fs, 0.012);
            let gm = match name {
                "windscreen" => mat::SCREEN,
                "backlight" => mat::PRIVACY,
                _ => mat::GLASS,
            };
            // Nearly flush (3 mm in): the frame and seal hide the step.
            let mut g = trim::offset(&c.mesh, -0.003, false, Some(gm), 1.0);
            if name == "glass_side" {
                // Rear doors' and quarter lights' glass: privacy tint.
                for k in 0..g.len() {
                    if g.pos[k].z < st.b_cut[0] {
                        g.mat[k] = mat::PRIVACY;
                    }
                }
            }
            g.smooth.iter_mut().for_each(|s| *s = 1);
            if frit > 0.0 {
                // The black frit: printed on the inside of the glass's border.
                let edge = trim::clip(m, &fs, 0.0);
                for (l, closed) in &edge.loops {
                    let fm = trim::loop_strip(
                        &edge.mesh,
                        l,
                        *closed,
                        frit,
                        0.012,
                        -0.0075,
                        mat::TRIM,
                        0.6,
                    );
                    g.append(&fm);
                }
            }
            let nm = if name == "glass_side" {
                format!("{name}_{}", if sx > 0.0 { "l" } else { "r" })
            } else {
                name.to_string()
            };
            // Tempered glass (side windows, backlight) is one rigid pane: it
            // shatters whole when its frame strains; the laminated windscreen
            // follows its frame and cracks.
            pieces.push(Piece {
                name: nm,
                mesh: g,
                bind: if name == "windscreen" {
                    Bind::Vote
                } else {
                    Bind::Rigid
                },
            });
        }
    }
    // Black window frames (strips along each side window's cut, so they follow
    // it exactly) and the B-pillar appliques.
    let gs = metric(m, &trim::field(m, glass_side));
    let bp = metric(m, &trim::field(m, b_pillar));
    for sx in [1.0f32, -1.0] {
        let fs: Vec<f32> = gs
            .iter()
            .zip(&m.pos)
            .map(|(v, p)| v.max(-p.x * sx))
            .collect();
        let c = trim::clip(m, &fs, 0.0);
        let mut mesh = Mesh::default();
        for (l, closed) in &c.loops {
            let mut s =
                trim::loop_strip(&c.mesh, l, *closed, 0.003, st.frame, 0.001, mat::TRIM, 1.0);
            let loops = trim::boundary_loops(&s);
            for (l2, c2) in &loops {
                trim::roll_loop(&mut s, l2, *c2, EdgeStyle::TRIM);
            }
            mesh.append(&s);
        }
        // The B-pillar applique in two halves, one on each door's frame (split
        // at the shut line), so each rides its own door.
        for (half, k) in [("f", 1.0f32), ("r", -1.0)] {
            let bs: Vec<f32> = bp
                .iter()
                .zip(&m.pos)
                .map(|(v, p)| v.max(-p.x * sx).max(k * (b_z(p.y) - p.z) + 0.0025))
                .collect();
            let b = trim::clip(m, &bs, 0.0);
            if b.mesh.is_empty() {
                continue;
            }
            let mut bm = trim::offset(&b.mesh, 0.0015, false, Some(mat::PILLAR), 1.0);
            let loops = trim::boundary_loops(&bm);
            for (l, closed) in &loops {
                trim::roll_loop(&mut bm, l, *closed, EdgeStyle::TRIM);
            }
            pieces.push(Piece {
                name: format!("b_pillar_{half}_{}", if sx > 0.0 { "l" } else { "r" }),
                mesh: bm,
                bind: Bind::Vote,
            });
        }
        pieces.push(Piece {
            name: format!("frame_{}", if sx > 0.0 { "l" } else { "r" }),
            mesh,
            bind: Bind::Vote,
        });
    }
    // --- Trim on the body: plate, tail garnish, belt mouldings ---------------
    let strip = |name: &str,
                 f: &[f32],
                 lift: f32,
                 mtl: u8,
                 roll: EdgeStyle,
                 sided: bool,
                 pieces: &mut Vec<Piece>| {
        let sides: &[f32] = if sided { &[1.0, -1.0] } else { &[0.0] };
        for &sx in sides {
            let fs: Vec<f32> = if sx == 0.0 {
                f.to_vec()
            } else {
                f.iter()
                    .zip(&m.pos)
                    .map(|(v, p)| v.max(-p.x * sx))
                    .collect()
            };
            let c = trim::clip(m, &fs, 0.0);
            if c.mesh.is_empty() {
                continue;
            }
            let mut mesh = trim::offset(&c.mesh, lift, false, Some(mtl), 1.0);
            mesh.smooth.iter_mut().for_each(|x| *x = 1);
            let loops = trim::boundary_loops(&mesh);
            for (l, closed) in &loops {
                trim::roll_loop(&mut mesh, l, *closed, roll);
            }
            pieces.push(Piece {
                name: name.to_string(),
                mesh,
                bind: Bind::Vote,
            });
        }
    };
    let plate_f = metric(
        m,
        &trim::field(m, |i, p| {
            if !on_band(i) {
                return 1.0;
            }
            let (a, _) = rear_ah(i, p);
            smax(a - 0.265, (p.y - 0.565).abs() - 0.064, 0.025)
        }),
    );
    let plate_back: Vec<f32> = plate_f.iter().map(|v| v - 0.008).collect();
    strip(
        "plate_surround",
        &plate_back,
        0.001,
        mat::TRIM,
        EdgeStyle::TRIM,
        false,
        &mut pieces,
    );
    strip(
        "plate",
        &plate_f,
        0.006,
        mat::PLATE,
        EdgeStyle {
            gap: 0.0,
            roll: 0.0015,
            depth: 0.005,
            flange: 0.0,
            ao_floor: 0.4,
            seal: false,
        },
        false,
        &mut pieces,
    );
    // The front plate (optional: NaN height for none): on its bracket.
    if st.plate_front_h.is_finite() {
        let plate_front = metric(
            m,
            // Continuous everywhere (a jump in the field would be made metric into
            // a sliver at the jump).
            &trim::field(m, |i, p| {
                let (a, h) = front_ah(i, p);
                smax(a.abs() - 0.262, (h - st.plate_front_h).abs() - 0.058, 0.02)
                    + if p.z < prof.length - 0.6 { 1.0 } else { 0.0 }
            }),
        );
        let plate_front_back: Vec<f32> = plate_front.iter().map(|v| v - 0.01).collect();
        strip(
            "plate_front_surround",
            &plate_front_back,
            0.016,
            mat::TRIM,
            EdgeStyle::TRIM,
            false,
            &mut pieces,
        );
        strip(
            "plate_front",
            &plate_front,
            0.021,
            mat::PLATE,
            EdgeStyle {
                gap: 0.0,
                roll: 0.0015,
                depth: 0.005,
                flange: 0.0,
                ao_floor: 0.4,
                seal: false,
            },
            false,
            &mut pieces,
        );
    }
    // The plates' texture coordinates (their lettering): u across as seen
    // from outside the car, v up.
    for pc in pieces.iter_mut() {
        let front = match pc.name.as_str() {
            "plate" => false,
            "plate_front" => true,
            _ => continue,
        };
        let m = &mut pc.mesh;
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in &m.pos {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        for k in 0..m.len() {
            let p = m.pos[k];
            let u = (p.x - lo.x) / (hi.x - lo.x).max(1e-4);
            let v = (p.y - lo.y) / (hi.y - lo.y).max(1e-4);
            m.uv[k] = [if front { u } else { 1.0 - u }, v];
        }
    }
    // Belt mouldings: chrome strips swept along the surface.
    for sx in [1.0f32, -1.0] {
        let mut belt = Mesh::default();
        let mut path = Vec::new();
        let (z0, z1) = (st.door_rear[0] + 0.05, st.door_front[0] - 0.04);
        for k in 0..=80 {
            let z = z0 + (z1 - z0) * k as f32 / 80.0;
            let a = surf.arc_at_z(z);
            let y = surf.band_top_y(a) - 0.005;
            let (p, n) = surf.band_at(a, y, sx);
            path.push(p + n * 0.002);
        }
        belt.tube(&path, Vec3::X * sx, 0.006, 0.0018, 10, mat::CHROME, 1.0);
        pieces.push(Piece {
            name: "belt".into(),
            mesh: belt,
            bind: Bind::Vote,
        });
    }
    // --- Fixtures ----------------------------------------------------------
    let avg_in = |f: &[f32]| -> Vec3 {
        let mut a = Vec3::ZERO;
        for (k, v) in f.iter().enumerate() {
            if *v < 0.0 {
                a -= m.nrm[k];
            }
        }
        a.normalize_or(-Vec3::Z)
    };
    let sided = |f: &[f32], sx: f32| -> Vec<f32> {
        f.iter()
            .zip(&m.pos)
            .map(|(v, p)| v.max(-p.x * sx))
            .collect()
    };
    // Headlights.
    let hf = metric(m, &trim::field(m, headlamp));
    for sx in [1.0f32, -1.0] {
        let f = sided(&hf, sx);
        let axis = (avg_in(&f) * 0.6 - Vec3::Z * 0.4).normalize();
        let projectors = [0.50f32, 0.64]
            .iter()
            .map(|&a| {
                let y = surf.band_top_y(a) - 0.036;
                surf.band_at(a, y, sx).0
            })
            .collect();
        let spec = fixtures::LampSpec {
            axis,
            depth: 0.09,
            lens: mat::LENS,
            housing: mat::TRIM,
            reflector: mat::TRIM,
            guides: vec![(-0.0105, -0.0042, mat::DRL)],
            guide_depth: 0.006,
            projectors,
            projector_r: 0.019,
            guide_above: (0..=24)
                .map(|k| {
                    let a = 0.36 + 0.80 * k as f32 / 24.0;
                    let mid = -0.040 + 0.022 * ((a - 0.40) / 0.70).clamp(0.0, 1.0);
                    surf.band_at(a, surf.band_top_y(a) + mid, sx).0
                })
                .collect(),
            lens_bulge: 0.004,
            bars: Vec::new(),
            bar_half: 0.0,
        };
        pieces.push(Piece {
            name: format!("headlight_{}", if sx > 0.0 { "l" } else { "r" }),
            mesh: fixtures::lamp(m, &f, &spec),
            bind: Bind::Rigid,
        });
    }
    // Tail lights.
    let tf = metric(m, &trim::field(m, taillamp));
    for sx in [1.0f32, -1.0] {
        let f = sided(&tf, sx);
        let axis = (avg_in(&f) * 0.6 + Vec3::Z * 0.4).normalize();
        let spec = fixtures::LampSpec {
            axis,
            depth: 0.09,
            lens: mat::TAIL_LENS,
            housing: mat::TRIM,
            reflector: mat::TAIL_BACK,
            guides: vec![
                (-0.010, -0.0045, mat::TAIL_LED),
                (-0.030, -0.0255, mat::TAIL_LED),
            ],
            guide_depth: 0.008,
            projectors: Vec::new(),
            projector_r: 0.0,
            guide_above: Vec::new(),
            // A swelled ruby lens over stacked light blades.
            lens_bulge: 0.008,
            bars: vec![0.22, 0.4, 0.58],
            bar_half: 0.0035,
        };
        pieces.push(Piece {
            name: format!("taillight_{}", if sx > 0.0 { "l" } else { "r" }),
            mesh: fixtures::lamp(m, &f, &spec),
            bind: Bind::Rigid,
        });
    }
    let lb = metric(m, &trim::field(m, lightbar));
    let spec = fixtures::LampSpec {
        axis: Vec3::Z,
        depth: 0.03,
        lens: mat::TAIL_LENS,
        housing: mat::TRIM,
        reflector: mat::TRIM,
        // A smoked full-width band: an LED outline and one blade through it.
        guides: vec![(-0.0065, -0.0035, mat::TAIL_LED)],
        guide_depth: 0.006,
        projectors: Vec::new(),
        projector_r: 0.0,
        guide_above: Vec::new(),
        lens_bulge: 0.003,
        bars: vec![0.5],
        bar_half: 0.0025,
    };
    pieces.push(Piece {
        name: "lightbar".into(),
        mesh: fixtures::lamp(m, &lb, &spec),
        bind: Bind::Rigid,
    });
    // Fog lamps: a slim clear lens with an LED line, in the bumper's corners.
    let ff = metric(m, &trim::field(m, foglamp));
    for sx in [1.0f32, -1.0] {
        if st.fog.is_empty() {
            break;
        }
        let f = sided(&ff, sx);
        let spec = fixtures::LampSpec {
            axis: (avg_in(&f) * 0.7 - Vec3::Z * 0.3).normalize(),
            depth: 0.04,
            lens: mat::LENS,
            housing: mat::TRIM,
            reflector: mat::REFLECTOR,
            guides: vec![(-0.009, -0.004, mat::DRL)],
            guide_depth: 0.006,
            projectors: Vec::new(),
            projector_r: 0.0,
            guide_above: Vec::new(),
            lens_bulge: 0.0,
            bars: Vec::new(),
            bar_half: 0.0,
        };
        pieces.push(Piece {
            name: format!("foglamp_{}", if sx > 0.0 { "l" } else { "r" }),
            mesh: fixtures::lamp(m, &f, &spec),
            bind: Bind::Rigid,
        });
    }
    // Grille and lower intake.
    let hfield: Vec<f32> = (0..m.len()).map(|i| m.pos[i].y - surf.ytop[i]).collect();
    let gf = metric(m, &trim::field(m, grille_f));
    let spec = fixtures::GrilleSpec {
        axis: -Vec3::Z,
        depth: 0.08,
        surround: 0.016,
        surround_mat: mat::TRIM,
        slats: Vec::new(),
        slat_half: 0.0075,
        slat_back: 0.018,
        back_mat: mat::GRILLE,
    };
    pieces.push(Piece {
        name: "grille".into(),
        mesh: fixtures::grille(m, &gf, &hfield, &spec),
        bind: Bind::Vote,
    });
    let inf = metric(m, &trim::field(m, intake_f));
    for sx in [1.0f32, -1.0] {
        let f = sided(&inf, sx);
        let spec = fixtures::GrilleSpec {
            axis: (avg_in(&f) * 0.5 - Vec3::Z * 0.5).normalize(),
            depth: 0.07,
            surround: 0.008,
            surround_mat: mat::TRIM,
            // Two gloss-black fins across the air curtain.
            slats: vec![(-0.212, mat::TRIM), (-0.252, mat::TRIM)],
            slat_half: 0.005,
            slat_back: 0.02,
            back_mat: mat::GRILLE,
        };
        pieces.push(Piece {
            name: format!("intake_{}", if sx > 0.0 { "l" } else { "r" }),
            mesh: fixtures::grille(m, &f, &hfield, &spec),
            bind: Bind::Vote,
        });
    }
    // Splitter: a gloss-black blade along the front bumper's lower edge,
    // standing 3 cm proud, from one corner to the other.
    {
        let mut path = Vec::new();
        let n_s = 60;
        for k in 0..=n_s {
            let a = -0.95 + 1.9 * k as f32 / n_s as f32;
            let th = surf.band.theta_front(a.abs());
            let sc = surf.band.section(th);
            let (p, n) = surf.band_at(a.abs(), sc.yl - 0.005, if a < 0.0 { -1.0 } else { 1.0 });
            let nh = Vec3::new(n.x, 0.0, n.z).normalize_or(Vec3::Z);
            path.push(p + nh * 0.004);
        }
        let mut sp = Mesh::default();
        sp.tube(&path, Vec3::Y, 0.03, 0.007, 12, mat::TRIM, 0.8);
        pieces.push(Piece {
            name: "splitter".into(),
            mesh: sp,
            bind: Bind::Vote,
        });
    }
    // Wheels, brakes, arch liners.
    for c in &def.corners {
        for sx in [1.0f32, -1.0] {
            let front = c.z > 0.5 * prof.length;
            let w = fixtures::WheelSpec {
                // Set 3 cm outboard of the physics tyre: the tyre's face
                // sits nearly flush with the arch, as a modern car's does.
                centre: Vec3::new(sx * (c.x + WHEEL_OUT), c.y, c.z),
                out: Vec3::X * sx,
                radius: c.tyre_radius,
                width: c.tyre_width,
                // A 20-inch look: a large rim on a low-profile sidewall.
                rim_radius: c.tyre_radius * 0.80,
                spokes: 10,
                disc_radius: if front { 0.188 } else { 0.176 },
            };
            pieces.push(Piece {
                name: "wheel".into(),
                mesh: fixtures::wheel(&w),
                bind: Bind::Wheel(u16::MAX),
            });
            pieces.push(Piece {
                name: "caliper".into(),
                mesh: fixtures::caliper(&w, if front { 0.7 } else { -0.9 }, -Vec3::Z),
                bind: Bind::Hub(u16::MAX),
            });
        }
    }
    for wl in &prof.wells {
        for sx in [1.0f32, -1.0] {
            let x_out = prof.half_width.eval(wl.z) - 0.02;
            pieces.push(Piece {
                name: "arch_liner".into(),
                mesh: fixtures::arch_liner(
                    Vec3::new(0.0, wl.y, wl.z),
                    sx,
                    st.arch_radius + 0.012,
                    x_out,
                    wl.inner_x - 0.05,
                ),
                bind: Bind::Vote,
            });
        }
    }
    // Mirrors and handles.
    for sx in [1.0f32, -1.0] {
        let z = st.door_front[0] - 0.12;
        let r = lidrow(z);
        let base = Vec3::new(sx * (r.x_t - 0.012), r.y_t + 0.03, z);
        pieces.push(Piece {
            name: "mirror".into(),
            mesh: fixtures::mirror(base, Vec3::X * sx, -Vec3::Z, mat::PAINT),
            bind: Bind::Rigid,
        });
        for zc in [st.b_cut[0] + 0.3, st.door_rear[0] + 0.24] {
            let mut path = Vec::new();
            let mut nrm = Vec::new();
            for k in 0..=12 {
                let z = zc + 0.09 - 0.18 * k as f32 / 12.0;
                let y = st.body.crease.eval(z) - 0.035;
                let a = surf.arc_at_z(z);
                let (p, n) = surf.band_at(a, y, sx);
                path.push(p);
                nrm.push(n);
            }
            pieces.push(Piece {
                name: "handle".into(),
                mesh: fixtures::handle(&path, &nrm, mat::CHROME),
                bind: Bind::Rigid,
            });
        }
    }
    // Underbody and exhaust.
    let mut rows = Vec::new();
    let mut z = 0.12;
    while z < prof.length - 0.15 {
        let th = surf.band.half - surf.arc_at_z(z);
        let edge = surf.band.point(th, 0.0);
        let mut w = edge.x - 0.005;
        for wl in &prof.wells {
            if (z - wl.z).abs() < st.arch_radius + 0.02 {
                w = w.min(wl.inner_x - 0.06);
            }
        }
        rows.push((z, edge.y + 0.004, w.max(0.1)));
        z += 0.05;
    }
    pieces.push(Piece {
        name: "underbody".into(),
        mesh: fixtures::underbody(&rows),
        bind: Bind::Vote,
    });
    for sx in [1.0f32, -1.0] {
        // The tip sits just inside its cut-out in the valance.
        let a = surf.band.half - 0.0;
        let _ = a;
        let x = sx * st.exhaust[0];
        let y = st.exhaust[1];
        let mut tip_z = 0.0f32;
        for k in 0..200 {
            // Rear surface z at (x, y): march the tail's band points.
            let ar = 0.2 + 0.005 * k as f32;
            let th = ar.min(surf.band.half);
            let sc = surf.band.section(th);
            let v = surf.band.v_at(&sc, y);
            let q = surf.band.point_in(&sc, v);
            if q.x >= st.exhaust[0] {
                tip_z = q.z;
                break;
            }
        }
        pieces.push(Piece {
            name: "exhaust".into(),
            mesh: fixtures::exhaust(
                Vec3::new(sx * 0.3, 0.21, 1.25),
                Vec3::new(x, y, tip_z + 0.012),
                -Vec3::Z,
                st.exhaust[2],
            ),
            bind: Bind::Vote,
        });
    }
    // Diffuser: vertical fins under the rear valance, between the tips.
    {
        let mut d = Mesh::default();
        for x in [-0.30f32, -0.15, 0.0, 0.15, 0.30] {
            // Each fin follows the tail's underside, 3 cm deep.
            let path: Vec<Vec3> = (0..=8)
                .map(|k| {
                    let z = 0.09 + 0.40 * k as f32 / 8.0;
                    Vec3::new(x, prof.bottom.eval(z) - 0.012, z)
                })
                .collect();
            d.tube(&path, Vec3::Y, 0.0035, 0.016, 8, mat::TRIM, 0.5);
        }
        pieces.push(Piece {
            name: "diffuser".into(),
            mesh: d,
            bind: Bind::Vote,
        });
    }
    // --- Cabin trim that follows the panels -----------------------------------
    // Door cards: each door's skin below the belt, 10 cm in, facing the cabin;
    // dark above the armrest, lighter below, with an armrest along it.
    for (name, region, _, _) in &panels {
        if !name.starts_with("door") {
            continue;
        }
        let f = metric(
            m,
            &trim::field(m, |i, p| {
                smax(
                    smax(region(i, p), p.y - (ytl(p) - 0.012), 0.01),
                    st.sill + 0.03 - p.y,
                    0.01,
                )
            }),
        );
        for sx in [1.0f32, -1.0] {
            let fs: Vec<f32> = f
                .iter()
                .zip(&m.pos)
                .map(|(v, p)| v.max(-p.x * sx))
                .collect();
            let c = trim::clip(m, &fs, -0.02);
            if c.mesh.is_empty() {
                continue;
            }
            let mut card = trim::offset(&c.mesh, -0.1, true, Some(mat::DOORCARD), 0.7);
            for k in 0..card.len() {
                if card.pos[k].y > 0.78 {
                    card.mat[k] = mat::DASH;
                }
            }
            let (mut z0, mut z1, mut xs) = (f32::MAX, f32::MIN, 0.0f32);
            for p in &card.pos {
                z0 = z0.min(p.z);
                z1 = z1.max(p.z);
                xs = xs.max(p.x.abs());
            }
            let arm: Vec<Vec3> = (0..=8)
                .map(|k| {
                    Vec3::new(
                        sx * (xs - 0.035),
                        0.66,
                        z0 + 0.05 + (z1 - z0 - 0.1) * k as f32 / 8.0,
                    )
                })
                .collect();
            card.tube(&arm, Vec3::Y, 0.03, 0.022, 12, mat::DASH, 0.6);
            pieces.push(Piece {
                name: format!("{name}_card_{}", if sx > 0.0 { "l" } else { "r" }),
                mesh: card,
                bind: Bind::Vote,
            });
        }
    }
    // Headliner and pillar trim: the greenhouse's skin (outside the glass)
    // 1.8 cm in, facing the cabin.
    let glass_all =
        |i: usize, p: Vec3| -> f32 { windscreen(i, p).min(backlight(i, p)).min(glass_side(i, p)) };
    let hl = metric(
        m,
        &trim::field(m, |i, p| {
            if !surf.lid[i] {
                return 1.0;
            }
            let cab = smax(st.backlight[0] - p.z, p.z - st.windscreen[1], 0.02);
            smax(
                smax(cab, ytl(p) + 0.025 - p.y, 0.02),
                -glass_all(i, p) - 0.004,
                0.005,
            )
        }),
    );
    let c = trim::clip(m, &hl, 0.0);
    if !c.mesh.is_empty() {
        pieces.push(Piece {
            name: "headliner".into(),
            mesh: trim::offset(&c.mesh, -0.018, true, Some(mat::HEADLINER), 0.8),
            bind: Bind::Vote,
        });
    }
    // Floor carpet and the parcel shelf (with the high-mounted stop lamp).
    let mut floor = Mesh::default();
    floor.superellipsoid(
        Vec3::new(0.0, 0.25, 2.35),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.78, 0.02, 1.0),
        10.0,
        super::segs(8, 3),
        mat::CARPET,
        0.45,
    );
    pieces.push(Piece {
        name: "carpet".into(),
        mesh: floor,
        bind: Bind::Vote,
    });
    let mut shelf = Mesh::default();
    let shelf_y = (prof.top.eval(st.backlight[0]) - 0.07).min(1.04);
    shelf.superellipsoid(
        Vec3::new(0.0, shelf_y, 0.5 * (st.backlight[0] + 1.34)),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.66, 0.012, 0.5 * (1.34 - st.backlight[0])),
        10.0,
        super::segs(6, 3),
        mat::TRIM,
        0.5,
    );
    // The high-mounted stop lamp: a slim LED bar inside the backlight's top
    // edge, just under the frit (it reads through the glass).
    {
        let zc = st.backlight[1] - 0.085;
        let t = Vec3::new(
            0.0,
            prof.top.eval(zc + 0.01) - prof.top.eval(zc - 0.01),
            0.02,
        )
        .normalize();
        let n = Vec3::X.cross(t).normalize();
        let n = if n.y < 0.0 { -n } else { n };
        let c = Vec3::new(0.0, prof.top.eval(zc), zc) - n * 0.03;
        shelf.superellipsoid(
            c,
            Vec3::X,
            n,
            t,
            Vec3::new(0.17, 0.012, 0.018),
            8.0,
            super::segs(6, 3),
            mat::TRIM,
            0.7,
        );
        shelf.superellipsoid(
            c + n * 0.011,
            Vec3::X,
            n,
            t,
            Vec3::new(0.155, 0.003, 0.006),
            8.0,
            super::segs(6, 3),
            mat::TAIL_LED,
            1.0,
        );
    }
    pieces.push(Piece {
        name: "parcel_shelf".into(),
        mesh: shelf,
        bind: Bind::Vote,
    });
    // --- Wipers, parked along the windscreen's base ----------------------------
    let glass_at = |x: f32, z: f32| -> (Vec3, Vec3) {
        // The nearest lid vertex in plan.
        let mut best = (f32::MAX, Vec3::ZERO, Vec3::Y);
        for k in 0..m.len() {
            if !surf.lid[k] {
                continue;
            }
            let p = m.pos[k];
            let d = (p.x - x).powi(2) + (p.z - z).powi(2);
            if d < best.0 {
                best = (d, p, m.nrm[k]);
            }
        }
        (Vec3::new(x, best.1.y, z), best.2)
    };
    let zw = st.windscreen[1] - 0.075;
    for (x0, x1, pivot_x) in [(-0.07f32, 0.57f32, 0.3f32), (-0.62, -0.06, -0.24)] {
        let mut w = Mesh::default();
        let n_b = 10;
        let blade: Vec<Vec3> = (0..=n_b)
            .map(|k| {
                let x = x0 + (x1 - x0) * k as f32 / n_b as f32;
                let z = zw + 0.05 * (x - x0) / (x1 - x0);
                let (p, n) = glass_at(x, z);
                p + n * 0.014
            })
            .collect();
        let (_, nm) = glass_at(0.5 * (x0 + x1), zw);
        w.tube(&blade, nm, 0.0045, 0.009, 8, mat::TRIM, 0.7);
        let (pv, pn) = glass_at(pivot_x, st.windscreen[1] + 0.02);
        let mid = blade[n_b / 2] + nm * 0.012;
        w.tube(&[pv + pn * 0.02, mid], nm, 0.006, 0.004, 8, mat::TRIM, 0.7);
        pieces.push(Piece {
            name: "wiper".into(),
            mesh: w,
            bind: Bind::Vote,
        });
    }
    // --- Emblems (an original mark: a split hexagon) ---------------------------
    {
        let a = 0.0;
        let (pf, nf) = surf.band_at(a, surf.band_top_y(a) - 0.042, 1.0);
        pieces.push(Piece {
            name: "emblem_front".into(),
            mesh: fixtures::emblem(pf + nf * 0.004, nf, Vec3::Y, 0.055),
            bind: Bind::Vote,
        });
        let ar = half;
        // On the light band's centre, proud of its lens.
        let (pr, nr) = surf.band_at(ar, surf.band_top_y(ar) + st.lightbar[1], 1.0);
        pieces.push(Piece {
            name: "emblem_rear".into(),
            mesh: fixtures::emblem(pr + nr * 0.005, nr, Vec3::Y, 0.058),
            bind: Bind::Vote,
        });
    }
    let _ = smin;
    bake_ao(&mut pieces, def, st);
    pieces
}

/// Ambient occlusion the car casts on itself, from analytic occluders (cheap,
/// baked into the vertices; the wheels turn, so the renderer shades theirs):
/// the floor under the car (downward faces low down), and the wheel arches
/// (faces inside the arch, round the tyre).
fn bake_ao(pieces: &mut [Piece], def: &VehicleDef, st: &Styling) {
    let prof = &def.profile;
    for pc in pieces.iter_mut() {
        if matches!(pc.bind, Bind::Wheel(_) | Bind::Hub(_)) {
            continue;
        }
        let m = &mut pc.mesh;
        for k in 0..m.len() {
            let (p, n) = (m.pos[k], m.nrm[k]);
            let mut occ = 1.0f32;
            // Under the car: a face looking down sees the floor in the car's
            // shadow; the lower it is, the darker.
            if n.y < 0.0 {
                let low = 1.0 - super::smoothstep(0.18, 0.62, p.y);
                occ *= 1.0 - 0.6 * (-n.y).min(1.0) * low;
            }
            // Inside a wheel arch: faces within the arch circle that look
            // inward or down (liners, the arch's return, the bumper's ends).
            for w in &prof.wells {
                let r = glam::Vec2::new(p.z - w.z, p.y - w.y).length();
                if r > st.arch_radius + 0.12 || p.x.abs() < w.inner_x - 0.12 {
                    continue;
                }
                let inward = (-n.x * p.x.signum()).max(0.0) + (-n.y).max(0.0) * 0.5;
                let inside = super::smoothstep(st.arch_radius + 0.10, st.arch_radius - 0.02, r);
                occ *= 1.0 - 0.55 * inside * inward.min(1.0);
            }
            m.ao[k] *= occ;
        }
    }
}

/// Roll a loop with a per-vertex style (arch lips vs shut lines): split it into
/// runs of one style.
fn roll_mixed(
    m: &mut Mesh,
    l: &[u32],
    closed: bool,
    style_at: &dyn Fn(Vec3) -> EdgeStyle,
    _d: EdgeStyle,
) {
    let n = l.len();
    // Style class per vertex: 0 shut line, 1 arch lip, 2 window seal.
    let class = |s: EdgeStyle| -> u8 {
        if s.roll > 0.005 {
            1
        } else if s.seal {
            2
        } else {
            0
        }
    };
    let style_of = |c: u8| -> EdgeStyle {
        match c {
            1 => EdgeStyle::ARCH,
            2 => EdgeStyle {
                seal: true,
                ..EdgeStyle::SHUT
            },
            _ => EdgeStyle::SHUT,
        }
    };
    let styles: Vec<u8> = l
        .iter()
        .map(|&v| class(style_at(m.pos[v as usize])))
        .collect();
    if styles.iter().all(|&s| s == styles[0]) {
        trim::roll_loop(m, l, closed, style_of(styles[0]));
        return;
    }
    // Rotate so a run starts at 0 (closed loops), then roll each run (sharing
    // its end vertex with the next, so the edge is continuous).
    let start = if closed {
        (0..n)
            .find(|&k| styles[k] != styles[(k + n - 1) % n])
            .unwrap_or(0)
    } else {
        0
    };
    let ord: Vec<usize> = (0..n).map(|k| (start + k) % n).collect();
    let mut run = vec![l[ord[0]]];
    let mut cur = styles[ord[0]];
    for &k in ord.iter().skip(1) {
        run.push(l[k]);
        if styles[k] != cur {
            trim::roll_loop(m, &run, false, style_of(cur));
            run = vec![l[k]];
            cur = styles[k];
        }
    }
    if closed {
        run.push(l[ord[0]]);
    }
    trim::roll_loop(m, &run, false, style_of(cur));
}

/// Make a field metric on the surface: divide by its gradient's length (per
/// vertex, from the triangles' linear gradients), so cut levels are distances.
pub fn metric(m: &Mesh, f: &[f32]) -> Vec<f32> {
    let mut g = vec![0.0f32; m.len()];
    let mut w = vec![0.0f32; m.len()];
    for t in m.idx.chunks(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let e1 = m.pos[b] - m.pos[a];
        let e2 = m.pos[c] - m.pos[a];
        let n = e1.cross(e2);
        let n2 = n.length_squared();
        if n2 < 1e-16 {
            continue;
        }
        let grad = ((f[b] - f[a]) * e2.cross(n) + (f[c] - f[a]) * n.cross(e1)) / n2;
        let gl = grad.length();
        if !gl.is_finite() {
            continue;
        }
        for i in [a, b, c] {
            g[i] += gl;
            w[i] += 1.0;
        }
    }
    f.iter()
        .enumerate()
        .map(|(i, v)| {
            let gl = if w[i] > 0.0 { g[i] / w[i] } else { 1.0 };
            v / gl.clamp(0.2, 50.0)
        })
        .collect()
}
