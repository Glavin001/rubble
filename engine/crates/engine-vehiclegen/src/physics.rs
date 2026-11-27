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

/// The front crash structure's rules (`engine/docs/sedan-front-audit.md` §6, the six changes
/// the owner approved on 2026-11-27; derivations and measurements in
/// `engine/docs/generated-vehicles-lessons.md` §25). Each is a named switch so a vehicle can
/// be generated with any combination and compared with the car before it: with every switch
/// off the generated files are byte for byte the earlier ones. A rule finds the members it
/// acts on by their names (the body-in-white's crash boxes and rails, the front subframe, the
/// closures): a design without them (a truck, a trailer) is unchanged.
///
/// In a design's physics block: `"front_rules": {"crash_box_stroke": true, ...}` (fields left
/// out are off).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FrontRules {
    /// §25.1: the front subframe's mounts land on the rail at the subframe's own corners, not
    /// on the crash boxes' rear nodes (whose beams spanned the rail's first bay at 157 kN each).
    pub subframe_mounts_off_crash_box: bool,
    /// §25.2: the crash box's bay carries the crash box alone: the shell's bracing beams that
    /// cross it (from the box's front nodes to the radiator support, rail, upper rail, towers)
    /// yield at a share of the box's own force ([`CRASH_BOX_BAY_BRACE_SHARE`]).
    pub crash_box_bay_alone: bool,
    /// §25.3: the crash box's chords crush flat (`deformLimitStress` = their yield) to
    /// `deformLimit` 0.73 ([`crate::thin_wall::crush::EFFECTIVE_CRUSH_FRACTION`]), or to the
    /// solver's least rest length where the chord is too short for that.
    pub crash_box_stroke: bool,
    /// §25.4: the SUPPORT stops and seals in the front crush zone (bumper cover, hood, fender
    /// seams) yield at the yield of the weakest sheet beam they bear on, instead of never.
    pub yielding_stops: bool,
    /// §25.5: a hood fold row near its front edge ([`HOOD_FOLD_ROW_M`] behind the front row),
    /// and the hood's and fenders' lengthwise beams at their panel's buckling force (Winter's
    /// effective width, [`crate::thin_wall::crush::winter_effective_width`]).
    pub hood_fold_row: bool,
    /// §25.6: in the `real_crash` character (front multiple 1), the crush members' chords at
    /// their section's mean crush force times the crash-rate factor at [`CRASH_RATE_SPEED`]
    /// ([`crate::thin_wall::crush::dynamic_factor`], the Camry deck's constants); the other
    /// characters keep their own multiple.
    pub crush_from_section: bool,
    /// §26: the crash box's lattice cell is a closed box: a node at each corner of its section
    /// at both ends (four more shell nodes a side, each end's nodes sharing its group's mass),
    /// chords at the four corners, every face braced, its axial yield its section's mean crush
    /// force. Off: the planar four-node cell (it swings about its rear nodes, §25.2).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub crash_box_cell: bool,
    /// The front load path designed from the 2012 Camry's (`engine/docs/front-load-path.md`):
    /// the crash boxes and rails closed four-corner cells back to the firewall, each bay at its
    /// section's mean crush force with a flat stroke; the bumper beam on a bolted plate on the
    /// boxes' front corners; the subframe mounted under the rail stations above its corners;
    /// the upper rails their own member to the hinge pillar; nothing else across the bays at
    /// more than [`CRASH_BOX_BAY_BRACE_SHARE`] of a bay; the engine's stops engaging where the
    /// block meets the structure. It calls rules 1, 4 and 5 and supersedes 2, 3 and the box
    /// cell ([`FrontRules::resolved`]). Off: every design's files as before.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub load_path: bool,
    /// With [`Self::load_path`]: the load path's bays crush by Rubble's crush law
    /// (`engine/docs/beam-yield-law.md`; CPU reference only): each bay's four chords one crush cell
    /// (peak, plateau, drop from the section, its collapse moments kept while it folds), its
    /// diagonals flat, every stroke to `deformLimit` past the solver's 0.1 m. Off: as before.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub crush_law: bool,
}

/// [`FrontRules::crash_box_bay_alone`]: the braces across a crash box's bay together carry
/// this share of the box's own chords' yield, in their axial direction. Engineering estimate
/// (the sheet-metal brackets between a crash box's front flange and the radiator support or
/// the upper rail fold at a small fraction of the box's force; a tenth is our estimate, not a
/// measurement); see the lessons doc §25.2.
pub const CRASH_BOX_BAY_BRACE_SHARE: f64 = 0.10;
/// [`FrontRules::hood_fold_row`]: the fold row's distance behind the hood's front row (m), the
/// middle of the 0.15–0.25 m where a hood's crush beads are pressed (audit §6.5; engineering
/// estimate).
pub const HOOD_FOLD_ROW_M: f32 = 0.20;
/// [`FrontRules::crush_from_section`]: the crush speed the crash-rate factor is taken at (m/s):
/// the NCAP full-frontal 56 km/h. The solver has no strain-rate law, so a rate has to be
/// chosen; the factor at a slower crash is lower (1.39 at 8 m/s against 1.52 at 15.6 m/s on the
/// crash box).
pub const CRASH_RATE_SPEED: f64 = 15.6;

impl FrontRules {
    /// Every rule off: the car as before the audit.
    pub fn is_off(&self) -> bool {
        *self == Self::default()
    }
    /// The rules a design takes when its physics block does not say: [`FrontRules::default`],
    /// or `RUBBLE_FRONT_RULES` when set (`all`, `none`, or rule names joined by commas), a
    /// measurement switch to run the gates and tests on another set without editing code (an
    /// unknown name panics, so a typo cannot pass as the default).
    pub fn from_env_or_default() -> Self {
        let Ok(v) = std::env::var("RUBBLE_FRONT_RULES") else { return Self::default() };
        let mut r = Self::default();
        for n in v.split(',').map(str::trim).filter(|n| !n.is_empty() && *n != "none") {
            match n {
                "all" => r = Self::all(),
                "subframe_mounts_off_crash_box" => r.subframe_mounts_off_crash_box = true,
                "crash_box_bay_alone" => r.crash_box_bay_alone = true,
                "crash_box_stroke" => r.crash_box_stroke = true,
                "yielding_stops" => r.yielding_stops = true,
                "hood_fold_row" => r.hood_fold_row = true,
                "crush_from_section" => r.crush_from_section = true,
                "crash_box_cell" => r.crash_box_cell = true,
                "load_path" => r.load_path = true,
                "crush_law" => r.crush_law = true,
                _ => panic!("RUBBLE_FRONT_RULES: no rule {n}"),
            }
        }
        r
    }
    /// Every rule on.
    pub fn all() -> Self {
        Self {
            subframe_mounts_off_crash_box: true,
            crash_box_bay_alone: true,
            crash_box_stroke: true,
            yielding_stops: true,
            hood_fold_row: true,
            crush_from_section: true,
            crash_box_cell: true,
            load_path: false,
            crush_law: false,
        }
    }
    /// The rules the rigger applies: with [`FrontRules::load_path`] on, the earlier rules it
    /// calls (1 the subframe's mounts, 4 the yielding stops, 5 the hood row and panels, 6 the
    /// crash rate) are on and those it supersedes (2 the bay's braces, 3 the box's stroke, the
    /// box cell: each bay's cell, braces and stroke are its own) are off. Without it, the rules
    /// as given.
    pub fn resolved(&self) -> Self {
        if !self.load_path {
            return self.clone();
        }
        Self {
            subframe_mounts_off_crash_box: load_path_step("mounts"),
            crash_box_bay_alone: false,
            crash_box_stroke: false,
            yielding_stops: load_path_step("panels"),
            hood_fold_row: load_path_step("panels"),
            crush_from_section: true,
            crash_box_cell: false,
            load_path: true,
            crush_law: self.crush_law,
        }
    }
}

/// Whether a step of [`FrontRules::load_path`] is on: all of them, unless `LOAD_PATH_STEPS` (a
/// measurement switch, comma separated: `cells`, `plate`, `mounts`, `upper`, `panels`, `engine`,
/// or `all`) lists the ones to build, so each step's effect can be measured alone.
pub fn load_path_step(name: &str) -> bool {
    match std::env::var("LOAD_PATH_STEPS") {
        Ok(v) => v.split(',').any(|s| s.trim() == name || s.trim() == "all"),
        Err(_) => true,
    }
}

/// Whether a measurement of [`FrontRules::crush_law`]'s cells is on (`CRUSH_CELL_EXP`, comma
/// separated; unset: none). Measurements only, never a default: each changes the cells against
/// their sections to show which part of the lattice keeps the rail from folding
/// (`engine/docs/beam-yield-law.md` §9): `chord_k` (the chords start the stiffness budget at 4 ×
/// their zone's spring, so they take more of their nodes' budget), `diag_tc` (the face diagonals
/// as strong in compression as in tension), `camry_sections` (the crash box and rail at the
/// Camry's crash zone 92 × 163 × 1.70 mm and mid rail 83 × 145 × 2.06 mm, `front-load-path.md`
/// §1.3), `no_peak` (the chords flat at the plateau: no first-buckle peak).
pub fn crush_cell_exp(name: &str) -> bool {
    std::env::var("CRUSH_CELL_EXP").is_ok_and(|v| v.split(',').any(|s| s.trim() == name))
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
    /// Size each raycast wheel's suspension for its own axle's static load (the same
    /// frequency and damping ratio on every wheel), not for the mean wheel load. Off by
    /// default (every car as before); a vehicle whose axles carry very different loads
    /// (a tractor unit's front axle: 1.39 × the mean) turns it on, or it parks
    /// nose-down past its design. Written to the JBeam as `rubbleSuspensionByLoad`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub suspension_by_load: bool,
    /// The front crash structure's rules ([`FrontRules`]).
    #[serde(skip_serializing_if = "FrontRules::is_off")]
    pub front_rules: FrontRules,
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
            suspension_by_load: false,
            front_rules: FrontRules::from_env_or_default(),
        }
    }
}
