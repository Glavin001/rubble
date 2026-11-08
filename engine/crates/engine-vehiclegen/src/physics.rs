//! A design's crash physics block: the masses, zone yields, hardware and
//! lattice reach the physics rigger (`engine-softbody`
//! `authoring::gen2::rig`, `engine/docs/generated-vehicles.md`) builds the
//! car's node-beam structure with. It lives with the designs so a parameter
//! set (the designer's) carries its physics; every field defaults to the
//! census / 2011 Accord value, so an empty block is a valid car.

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
        }
    }
}
