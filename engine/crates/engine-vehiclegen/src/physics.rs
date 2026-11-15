//! A design's crash physics block: the masses, zone yields, hardware and
//! lattice reach the physics rigger (`engine-softbody`
//! `authoring::gen2::rig`, `engine/docs/generated-vehicles.md`) builds the
//! car's node-beam structure with. It lives with the designs so a parameter
//! set (the designer's) carries its physics; every field defaults to the
//! census / 2011 Accord value, so an empty block is a valid car.

/// The crash character of a design: how strong its crush members and panels
/// are against the body's definitions (`structure`: sections, gauges, grades).
/// One setting, two documented presets, every number editable:
///
/// * [`CrashFeel::beamng_feel`] (the default): the look and feel of a good
///   BeamNG car (the ETK K-series, measured in the same scenes, same solver):
///   shallower crush, more spring-back, panels that stay in shape away from
///   the hit. Its front is stronger than the real crash bands.
/// * [`CrashFeel::real_crash`]: the members as defined (their own section,
///   gauge and grade), which is what the NHTSA full-frontal bands
///   (`engine/docs/crash-reference-targets.md`) are measured against.
///
/// Each number is a multiple of a member's mean crush force or of a panel's
/// yield strain, applied to the definitions before the lattice is built, so it
/// is a statement about the steel: a thin-walled box's mean crush force goes
/// as σ₀·t^(5/3), so a multiple `s` is the same section in a gauge `s^0.6`
/// times as thick ([`CrashFeel::gauge_ratio`]), or the same gauge in a grade
/// of `s` times the flow stress. Nothing is scaled beam by beam.
///
/// In a design's physics block: `"crash": "real_crash"` or
/// `"crash": {"front_crush": 1.4, ...}` (fields left out take the default
/// preset's).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(from = "CrashFeelIn")]
pub struct CrashFeel {
    /// Front crush members (crash boxes, front rails, radiator support,
    /// upper rails, the rails' and towers' bracing): mean crush force.
    pub front_crush: f32,
    /// Strut towers and the subframe's mount nodes on the rails.
    pub tower: f32,
    /// The cabin (firewall, cowl, floor, sills, pillars, roof).
    pub cabin: f32,
    /// Rear rails, boot floor, tail and quarter panels.
    pub rear_crush: f32,
    /// Bolt-on panels and closures (hood, fenders, doors, lid): yield strain
    /// of their sheet and frames.
    pub panel_yield: f32,
    /// Bumper covers: yield strain.
    pub bumper_yield: f32,
}

impl CrashFeel {
    /// The members as defined.
    pub fn real_crash() -> Self {
        Self { front_crush: 1.0, tower: 1.0, cabin: 1.0, rear_crush: 1.0, panel_yield: 1.0, bumper_yield: 1.0 }
    }
    /// The ETK's level (see `engine/docs/generated-vehicles.md`, "Crash
    /// character", for the side-by-side numbers these were set from).
    ///
    /// Front members at 1.8 times their defined mean crush force: the crash
    /// box's 1.5 mm and the rail's 1.8 mm HSLA as 2.1 mm and 2.6 mm, or the
    /// same gauges in a steel of 710 MPa flow stress (DP980 class). Towers and
    /// cabin at 1.5 (gauge × 1.28). Panels at twice the census yield strain
    /// (hood and fenders 6 %, doors 2.9 %), which is what keeps them in shape
    /// away from the hit. The rear is as defined (not yet measured against
    /// the ETK).
    pub fn beamng_feel() -> Self {
        Self { front_crush: 1.8, tower: 1.5, cabin: 1.5, rear_crush: 1.0, panel_yield: 2.0, bumper_yield: 1.0 }
    }
    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "beamng_feel" => Some(Self::beamng_feel()),
            "real_crash" => Some(Self::real_crash()),
            _ => None,
        }
    }
    /// The gauge that gives a box section `scale` times its mean crush force
    /// at the same grade, over its defined gauge.
    pub fn gauge_ratio(scale: f32) -> f32 {
        scale.powf(0.6)
    }
}

impl Default for CrashFeel {
    fn default() -> Self {
        Self::beamng_feel()
    }
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum CrashFeelIn {
    Preset(String),
    Table {
        front_crush: Option<f32>,
        tower: Option<f32>,
        cabin: Option<f32>,
        rear_crush: Option<f32>,
        panel_yield: Option<f32>,
        bumper_yield: Option<f32>,
    },
}

impl From<CrashFeelIn> for CrashFeel {
    fn from(v: CrashFeelIn) -> Self {
        let d = CrashFeel::default();
        match v {
            // (an unknown name is the default: `Physics` has no fallible path
            // here; `CrashFeel::preset` is the checked lookup)
            CrashFeelIn::Preset(n) => CrashFeel::preset(&n).unwrap_or(d),
            CrashFeelIn::Table { front_crush, tower, cabin, rear_crush, panel_yield, bumper_yield } => CrashFeel {
                front_crush: front_crush.unwrap_or(d.front_crush),
                tower: tower.unwrap_or(d.tower),
                cabin: cabin.unwrap_or(d.cabin),
                rear_crush: rear_crush.unwrap_or(d.rear_crush),
                panel_yield: panel_yield.unwrap_or(d.panel_yield),
                bumper_yield: bumper_yield.unwrap_or(d.bumper_yield),
            },
        }
    }
}

/// The editable physics of a design: masses, zone yields, hardware and the
/// lattice's reach. Every field has the census / 2011 Accord default gen3 was
/// calibrated with, so an empty block gives a valid car; a vehicle-gen
/// design states only what differs (`PHYSICS = {...}` in its design file,
/// exported in `rig.json`; `engine/docs/generated-vehicles.md`). Unknown keys
/// are an error, so a typo fails loudly.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Physics {
    /// Kerb mass with the driver (kg). What the parts and named masses do not
    /// account for goes to the front-end module.
    pub kerb: f32,
    /// Part masses (kg): body in white, each front and rear door, each
    /// fender, hood, decklid (or tailgate), engine, transaxle, bumper
    /// assemblies (beam and fascia).
    pub biw: f32,
    pub door_f: f32,
    pub door_r: f32,
    pub fender: f32,
    pub hood: f32,
    pub decklid: f32,
    pub engine: f32,
    pub transaxle: f32,
    pub bumper_f: f32,
    pub bumper_r: f32,
    /// Overrides of the named masses (kg) by name: driver, front_seats,
    /// rear_seat, dashboard_steering, fuel, interior, exhaust, spare_wheel,
    /// battery, radiator_cooling, windshield, rear_glass,
    /// front_subframe_steering, rear_subframe, hvac_wiring.
    pub named: std::collections::BTreeMap<String, f32>,
    /// The powertrain's faces ahead of and behind the front axle (m), the
    /// dash line behind the axle (m) and what stays crushed between them.
    pub pt_ahead: f32,
    pub pt_behind: f32,
    pub dash_behind: [f32; 2],
    pub pt_clearance: f32,
    /// Zone yields as multiples of the census yield (1.15 %), with the steel
    /// each stands for: the crush zones (DP600 rails and cans), the
    /// suspension towers, the footwell, the greenhouse, the A-pillar's chords
    /// (22MnB5), the floor's cross members, the floor pan (0.7 mm HC220B),
    /// the B-pillar above and below its tailored split, the doors' intrusion
    /// beams, the side below the belt, the doors.
    pub crush_yield: f32,
    pub tower_yield: f32,
    pub footwell_yield: f32,
    pub greenhouse_yield: f32,
    pub apillar_yield: f32,
    pub cross_member_yield: f32,
    pub floor_pan_yield: f32,
    pub bpillar_upper_yield: f32,
    pub bpillar_lower_yield: f32,
    pub intrusion_yield: f32,
    pub side_yield: f32,
    pub door_yield: f32,
    /// Hardware strengths per beam (N).
    pub door_hinge: f32,
    pub door_latch: f32,
    pub hood_hinge: f32,
    pub hood_latch: f32,
    pub trunk_hinge: f32,
    pub trunk_latch: f32,
    pub fender_bolt: f32,
    pub fender_row_bolt: f32,
    pub bumper_bolt: f32,
    pub bumper_side_bolt: f32,
    pub engine_mount: f32,
    /// Stacks (m): the cooling module crushed solid; a front wheel's hub to
    /// the hinge pillar once its tyre is crushed onto the rim.
    pub cooling_stack: f32,
    pub wheel_stack: f32,
    /// The beam graph's reach (m): the body's members, each closure's frame.
    pub reach_body: f32,
    pub reach_panel: f32,
    /// Shut-line edge nodes' spacing (m).
    pub edge_spacing: f32,
    /// The car's drag coefficient on its frontal area (the collision skin's
    /// triangle aerodynamics are set to give it: no per-node drag). 0.30: a
    /// current mid-size sedan (the 2011 Accord's published Cd is 0.30-0.31).
    pub drag_cd: f32,
    /// The crash character ([`CrashFeel`]): `"beamng_feel"` (default),
    /// `"real_crash"`, or its numbers. Used by cars on the body definitions
    /// (`structure`); the earlier lattices have the zone yields above.
    pub crash: CrashFeel,
}

impl Default for Physics {
    /// The census / 2011 Honda Accord values gen3 was calibrated with
    /// (`engine-softbody` `authoring::gen2::rig` and `authoring::sedan`, whose
    /// test keeps the two in step): kerb with driver; Accord BIW, closures
    /// and powertrain (DOT HS 811 666); bumper assemblies (§5.10.7); census
    /// zone yields; hardware per beam (an M6 8.8 bolt's 16 kN, ISO 898-1).
    fn default() -> Self {
        Self {
            kerb: 1450.0,
            biw: 339.0,
            door_f: 58.99 / 2.0,
            door_r: 47.46 / 2.0,
            fender: (147.0 - 58.99 - 47.46 - 17.90 - 12.37) / 2.0,
            hood: 17.90,
            decklid: 12.37,
            engine: 169.9,
            transaxle: 96.7,
            bumper_f: 7.96 + 2.5,
            bumper_r: 7.84 + 1.3,
            named: Default::default(),
            pt_ahead: 0.39,
            pt_behind: 0.12,
            dash_behind: [0.27, 0.30],
            pt_clearance: 0.03,
            crush_yield: 0.23,
            tower_yield: 0.4,
            footwell_yield: 3.0,
            greenhouse_yield: 0.38,
            apillar_yield: 3.0,
            cross_member_yield: 3.0,
            floor_pan_yield: 0.3,
            bpillar_upper_yield: 3.0,
            bpillar_lower_yield: 0.5,
            intrusion_yield: 1.0,
            side_yield: 1.0,
            door_yield: 1.0,
            door_hinge: 40e3,
            door_latch: 50e3,
            hood_hinge: 40e3,
            hood_latch: 30e3,
            trunk_hinge: 25e3,
            trunk_latch: 45e3,
            fender_bolt: 25e3,
            fender_row_bolt: 16e3,
            bumper_bolt: 20e3,
            bumper_side_bolt: 15e3,
            engine_mount: 60e3,
            cooling_stack: 0.08,
            wheel_stack: 0.23,
            reach_body: 0.90,
            reach_panel: 0.66,
            edge_spacing: 0.22,
            drag_cd: 0.30,
            crash: CrashFeel::default(),
        }
    }
}
