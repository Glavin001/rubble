//! Rigs of Rods `.truck` loader (phase-1 structure sections plus the driving
//! sections stage 1 needs). Clean-room: written from the documented file format
//! and observed behaviour, not from the GPL engine source.
//!
//! The parser keeps the file's state machine: block keywords open a section until
//! the next keyword, directives (`set_beam_defaults`, `detacher_group`, …) change
//! state for every following line, and `end` ends the file.

use crate::desc::*;
use glam::Vec3;
use std::collections::HashMap;

/// Command keys a trigger can hold (1…84).
pub const COMMAND_KEYS: usize = 84;

/// RoR's built-in beam defaults.
pub const DEFAULT_SPRING: f32 = 9.0e6;
pub const DEFAULT_DAMP: f32 = 12000.0;
pub const DEFAULT_DEFORM: f32 = 400_000.0;
pub const DEFAULT_BREAK: f32 = 1.0e6;
pub const DEFAULT_MINIMASS: f32 = 50.0;
pub const DEFAULT_COLLISION_RANGE: f32 = 0.02;
/// Support beams break past L·factor of extension unless the line overrides it.
pub const DEFAULT_SUPPORT_BREAK_FACTOR: f32 = 4.0;

#[derive(Clone, Copy, Debug)]
struct BeamDefaults {
    spring: f32,
    damp: f32,
    deform: f32,
    brk: f32,
    plastic: f32,
    /// `plastic` was set explicitly (affects the 100 kN floor).
    plastic_set: bool,
    /// A `set_beam_defaults` line set these (`user`), and `enable_advanced_deformation`
    /// came before that line (`advanced`): a user deform threshold below 400 000 is
    /// raised to it unless `advanced`.
    user: bool,
    advanced: bool,
}

impl Default for BeamDefaults {
    fn default() -> Self {
        Self {
            spring: DEFAULT_SPRING,
            damp: DEFAULT_DAMP,
            deform: DEFAULT_DEFORM,
            brk: DEFAULT_BREAK,
            plastic: 0.0,
            plastic_set: false,
            user: false,
            advanced: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Scale {
    spring: f32,
    damp: f32,
    deform: f32,
    brk: f32,
}

impl Default for Scale {
    fn default() -> Self {
        Self {
            spring: 1.0,
            damp: 1.0,
            deform: 1.0,
            brk: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct NodeDefaults {
    load_weight: Option<f32>,
    friction: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    None,
    Globals,
    Nodes,
    Nodes2,
    Beams,
    Shocks,
    Shocks2,
    Shocks3,
    Cab,
    Contacters,
    Wheels,
    Wheels2,
    MeshWheels,
    MeshWheels2,
    FlexBodyWheels,
    Engine,
    Brakes,
    Hydros,
    Commands,
    Commands2,
    Wings,
    FuseDrag,
    Fixes,
    Minimass,
    Cameras,
    Cinecam,
    Triggers,
    Animators,
    SlideNodes,
    TyrePose,
    SpawnContacts,
    ContactBreaks,
    JbeamPlastic,
    TensionYield,
    DampCutoff,
    AeroTri,
    Skipped,
}

/// A `cinecam`: a node (no ground contact) at `position` on 8 normal beams with the
/// beam defaults in force at its line, its own spring and damping, and its own mass
/// (raised to the minimum node mass), as RoR's `ProcessCinecam`.
#[derive(Clone, Debug)]
pub struct TruckCinecam {
    pub position: [f32; 3],
    pub nodes: [u32; 8],
    /// Template for its beams (spring, damping, thresholds; `a`, `b` unset).
    pub beam: SoftBeam,
    pub node_mass: f32,
}

/// A wheel as written in the file (before lowering to a raycast wheel).
#[derive(Clone, Debug)]
pub struct TruckWheel {
    pub radius: f32,
    pub width: f32,
    pub rays: u32,
    pub node1: u32,
    pub node2: u32,
    pub arm: Option<u32>,
    pub braked: u32,
    pub propulsed: u32,
    pub mass: f32,
    /// Section the wheel came from.
    pub kind: &'static str,
    /// `wheels` only: tyre spring and damping, the rigidity node, and the beam and
    /// node defaults in force at the line (Rigs of Rods' soft tyre, [`crate::ror`]).
    pub spring: f32,
    pub damp: f32,
    /// The rim beams' spring and damping (the 4 tread-ring beams per ray): the tyre's
    /// for `wheels` and `meshwheels`; for `meshwheels2` the beam defaults at the line,
    /// unscaled (`engine/docs/explicit-beams-wheels2-shocks2-spec.md` §2.4).
    pub rim_spring: f32,
    pub rim_damp: f32,
    pub rigidity: Option<u32>,
    pub beam: SoftBeam,
    pub node_friction: f32,
}

/// A `commands` / `commands2` line: a key-driven actuator beam. With no driver input
/// only self-centering lines move (`engine/docs/explicit-beams-aero-commands-spec.md`).
#[derive(Clone, Debug, PartialEq)]
pub struct TruckCommand {
    /// Its beam in [`TruckFile::beams`].
    pub beam: usize,
    /// Rest-length rates (m/s) toward the contracted and the extended limit.
    pub shorten_rate: f32,
    pub lengthen_rate: f32,
    /// Limits as multiples of the spawn length.
    pub max_contraction: f32,
    pub max_extension: f32,
    pub contract_key: i32,
    pub extend_key: i32,
    /// Option `r`: no stiffness when shorter than its rest length.
    pub rope: bool,
    /// Option `c` (when it comes before `o` and `p`): drives toward the centre of
    /// its limits with no input.
    pub centering: bool,
    /// Option `p` or `o` came first of `o`, `p`, `c` (one-press modes).
    pub one_press: bool,
    /// The line's key input has no inertia: its start and stop delays are absent or
    /// both 0, and no `set_inertia_defaults` with a positive delay came before it.
    pub no_inertia: bool,
}

/// A `triggers` line (`engine/docs/explicit-beams-sections-triggers-slidenodes-spec.md`
/// §2): a zero-force beam that holds a command key while its length is past a bound.
#[derive(Clone, Debug, PartialEq)]
pub struct TruckTrigger {
    /// Its beam in [`TruckFile::beams`] (a hydro-kind beam with no stiffness or
    /// damping, its gate in [`TruckFile::actuator_gates`]).
    pub beam: usize,
    /// Short and long bounds (fractions of the rest length) after option `c`'s
    /// conversion.
    pub short_bound: f32,
    pub long_bound: f32,
    /// Command keys (1…84) held on the short and the long side (long 0: none).
    pub short_key: u32,
    pub long_key: u32,
}

/// A `slidenodes` line (spec §3): a node held by an undamped spring to a rail of
/// beams.
#[derive(Clone, Debug, PartialEq)]
pub struct TruckSlideNode {
    pub node: u32,
    /// Rail nodes in the order written (ranges expanded).
    pub rail: Vec<u32>,
    /// Spring rate (N/m), break force (N), tolerance (m).
    pub spring: f32,
    pub break_force: f32,
    pub tolerance: f32,
}

/// A `wings` line: one aerodynamic panel on 8 node slots (front-left-down,
/// front-right-down, front-left-up, front-right-up, then the same at the back).
#[derive(Clone, Debug, PartialEq)]
pub struct TruckWing {
    pub nodes: [u32; 8],
    /// Control-surface letter (`n` none).
    pub control: char,
    pub chord_point: f32,
    pub min_deflection: f32,
    pub max_deflection: f32,
    /// Airfoil table file name (`.afl`).
    pub airfoil: String,
    pub efficacy: f32,
}

/// A `shocks2` line: a progressive shock with separate compression (in) and
/// extension (out) rates and hard or soft bump limits
/// (`engine/docs/explicit-beams-wheels2-shocks2-spec.md` §4–6). Its beam in
/// [`TruckFile::beams`] is a bounded beam with the "in" pair, for the solvers that
/// do not run the shocks2 force law.
#[derive(Clone, Debug, PartialEq)]
pub struct TruckShock2 {
    /// Its beam in [`TruckFile::beams`].
    pub beam: usize,
    pub spring_in: f32,
    pub damp_in: f32,
    pub prog_spring_in: f32,
    pub prog_damp_in: f32,
    pub spring_out: f32,
    pub damp_out: f32,
    pub prog_spring_out: f32,
    pub prog_damp_out: f32,
    /// Bounds as written (before the `m` / `M` conversions, which need the spawn
    /// positions).
    pub short_bound: f32,
    pub long_bound: f32,
    /// Precompression as written (0 is accepted).
    pub precompression: f32,
    /// Options `m` (metric bounds), `M` (absolute metric bounds), `s` (soft bump).
    pub metric: bool,
    pub abs_metric: bool,
    pub soft: bool,
    /// Bump spring and damping: the beam defaults at the line, unscaled.
    pub bump_spring: f32,
    pub bump_damp: f32,
    /// The deform threshold at the line (the break test's gate).
    pub gate: f32,
    /// 4 × the unscaled break threshold.
    pub strength: f32,
}

/// A beam end: a node of the file, or one a `wheels` line generates (its wheel in
/// [`TruckFile::wheels`] and its place in the rings, outer and inner alternating).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TruckNodeRef {
    File(u32),
    Tyre(u32, u32),
}

/// A `beams` line with an end on a generated tyre node (legacy numbering), which
/// only solvers that build the tyre nodes spawn.
#[derive(Clone, Debug)]
pub struct TruckTyreBeam {
    /// File beams ([`TruckFile::beams`]) before it.
    pub position: usize,
    pub a: TruckNodeRef,
    pub b: TruckNodeRef,
    /// Its parameters (`a`, `b` unset).
    pub beam: SoftBeam,
}

/// The `fusedrag` line (the last one of the section).
#[derive(Clone, Debug, PartialEq)]
pub struct TruckFuseDrag {
    pub front: u32,
    pub rear: u32,
    /// The fuselage width; with `autocalc`, the file nodes' z extent × y extent ×
    /// the area coefficient.
    pub width: f32,
    pub airfoil: String,
}

/// Parsed file before mass distribution, in file node indices.
#[derive(Clone, Debug, Default)]
pub struct TruckFile {
    pub title: String,
    pub file_format_version: u32,
    pub dry_mass: f32,
    pub cargo_mass: f32,
    pub minimass: f32,
    pub minimass_skip_loaded: bool,
    pub advanced_deformation: bool,
    pub collision_range: f32,
    /// The file sets `set_collision_range` (the BeamNG profile's band then yields
    /// to it).
    pub collision_range_set: bool,
    pub nodes: Vec<SoftNode>,
    /// Per node: loaded flag and optional load-weight override.
    pub loaded: Vec<(bool, Option<f32>)>,
    pub beams: Vec<SoftBeam>,
    pub triangles: Vec<SoftTriangle>,
    pub contacters: Vec<u32>,
    /// `rubble_full_contact` (Rubble's extension, spec §10.2): every node with
    /// collision is a contacter against other vehicles' triangles.
    pub full_contact: bool,
    /// `rubble_exact_lengths` (Rubble's extension): beam lengths through an exact
    /// square root, not RoR's fast inverse square root (its error, up to 0.18 % of a
    /// length, is comparable to metal's yield strain and far above glass's).
    pub exact_lengths: bool,
    /// `rubble_beamng_profile` (Rubble's extension, jbeam-semantics-spec.md §36):
    /// the vehicle runs the BeamNG profile's switches
    /// ([`crate::explicit_beams::ExplicitBeamsSettings::beamng`]) and two-body
    /// vehicle contact, as JBeam content does.
    pub beamng_profile: bool,
    /// `rubble_drag_coef <dragCoef> [<liftCoef>]` (Rubble's extension,
    /// jbeam-semantics-spec.md §36): the BeamNG profile's triangle aerodynamics
    /// with these coefficients on every cab triangle (BeamNG's defaults 100 and
    /// dragCoef otherwise; "around 10 for most exposed body panels").
    pub drag_coef: Option<(f32, f32)>,
    /// `rubble_rest_pose <y>` (Rubble's extension): the file's node positions are
    /// the vehicle at rest on flat ground at height `y` (its beams' rest lengths
    /// carry the unloaded geometry).
    pub rest_ground: Option<f32>,
    /// `rubble_tyre_pose` lines `wheel, node, x, y, z` (Rubble's extension): a
    /// generated tyre node's spawn position (the tyre at rest, deflected; its
    /// beams' rest lengths stay the undeflected tyre's). `wheel` is the file's
    /// wheel index, `node` the tyre's node (0 .. 2 × rays).
    pub tyre_pose: Vec<(u32, u32, [f32; 3])>,
    /// `rubble_tyre_pose` lines `wheel, x1, y1, z1, x2, y2, z2`: the tyre's
    /// axle nodes' rest positions (the unloaded design's), which its beams'
    /// rest lengths are built round.
    pub tyre_rest_axle: Vec<(u32, [f32; 3], [f32; 3])>,
    /// `rubble_spawn_contacts` lines `a, b, c, node` (Rubble's extension): a
    /// node and a collision triangle (by its nodes) that collide although the
    /// node starts in its range: in a file that is its vehicle at rest, they
    /// rest in contact (spec §10.1's spawn exclusions are the design's).
    pub spawn_contacts: Vec<([u32; 3], u32)>,
    /// `rubble_contact_breaks` lines `group, force_N, nodes…` (Rubble's extension,
    /// jbeam-semantics-spec.md §32): break group `group` goes off when the contact
    /// force on any of the nodes exceeds `force_N` in a step.
    pub contact_breaks: Vec<(u32, f32, Vec<u32>)>,
    /// `rubble_jbeam_plastic` lines `a, b, deformLimitStress_N, deformLimit,
    /// deformLimitExpansion` (Rubble's extension): the beam between nodes a and
    /// b yields by JBeam's rule (`jbeam-semantics-spec.md` §3) with those
    /// limits: its hardened thresholds never pass the stress cap, its rest
    /// length never shortens past (1 − deformLimit) or lengthens past
    /// (1 + deformLimitExpansion) of its spawn rest length (a negative value:
    /// no limit). It keeps the file's plastic coefficient.
    pub jbeam_plastic: Vec<(u32, u32, f32, f32, f32)>,
    /// `rubble_tension_yield` lines `a, b, Tt_N` (Rubble's extension,
    /// jbeam-semantics-spec.md §18.4): the deformable beam between nodes a and
    /// b yields in tension at Tt (its compression threshold stays the file's
    /// deform threshold): a sheet buckles in compression but carries its
    /// material's yield in tension.
    pub tension_yield: Vec<(u32, u32, f32)>,
    /// `rubble_damp_cutoff` lines `a, b, dampCutoffHz` (Rubble's extension,
    /// jbeam-semantics-spec.md §36): the BeamNG profile's beam a–b damps only
    /// its extension rate filtered at that cutoff (JBeam's `dampCutoffHz`, §2).
    pub damp_cutoff: Vec<(u32, u32, f32)>,
    /// `rubble_aero_tri` lines `a, b, c, dragCoef, liftCoef` (Rubble's extension,
    /// §36): the BeamNG profile's triangle aerodynamics per cab triangle (nodes in
    /// any order); with the section, an unlisted cab triangle has none.
    pub aero_tri: Vec<([u32; 3], f32, f32)>,
    pub wheels: Vec<TruckWheel>,
    pub engine: Option<SoftEngine>,
    pub brake_force: Option<f32>,
    pub cameras: Option<RefNodes>,
    /// `cinecam` lines: Rigs of Rods spawns a camera node there on 8 beams, which
    /// take part in the dry-mass split ([`distribute_masses`]) and are simulated.
    pub cinecams: Vec<TruckCinecam>,
    /// `commands` and `commands2` lines in file order.
    pub commands: Vec<TruckCommand>,
    /// The deform threshold in force at each hydro and command line (their beams
    /// in [`Self::beams`] carry an infinite `deform`: they never yield), by beam.
    pub actuator_gates: Vec<(usize, f32)>,
    pub wings: Vec<TruckWing>,
    pub fusedrag: Option<TruckFuseDrag>,
    /// `beams` lines on generated tyre nodes.
    pub tyre_beams: Vec<TruckTyreBeam>,
    /// `shocks2` lines in file order.
    pub shocks2: Vec<TruckShock2>,
    /// The beams ([`Self::beams`]) of `shocks3` lines.
    pub shocks3: Vec<usize>,
    /// `triggers` lines, `animators` lines (their beams, actuators of constant
    /// length) and `slidenodes` lines, in file order (root, then the selected
    /// section).
    pub triggers: Vec<TruckTrigger>,
    pub animators: Vec<usize>,
    pub slidenodes: Vec<TruckSlideNode>,
    /// Section configurations (`section` names) in byte order, and the one spawned
    /// with the root (none: the root alone).
    pub section_configs: Vec<String>,
    pub section_config: Option<String>,
    pub warnings: Vec<String>,
    /// Lines the explicit solver refuses (unsupported options): spawning fails with
    /// these.
    pub errors: Vec<String>,
    /// The last `submesh_groundmodel` line's surface name (the contact surface of
    /// the vehicle's collision triangles; none: concrete).
    pub submesh_groundmodel: Option<String>,
}

fn tokens(line: &str) -> Vec<&str> {
    line.split(|c: char| c == ',' || c == '|' || c == ':' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .collect()
}

fn f(t: &[&str], i: usize, default: f32) -> f32 {
    t.get(i)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(default)
}

/// `-1` (or any negative) means "use the default" in RoR's default directives.
fn fd(t: &[&str], i: usize, default: f32) -> f32 {
    let v = f(t, i, default);
    if v < 0.0 {
        default
    } else {
        v
    }
}

struct Parser {
    out: TruckFile,
    section: Section,
    defaults: BeamDefaults,
    scale: Scale,
    node_defaults: NodeDefaults,
    detacher_group: u32,
    /// File node id (or name) → node index.
    ids: HashMap<String, u32>,
    /// Next implicit number under the sequential importer.
    next_number: u32,
    skipped: HashMap<String, u32>,
    line_no: usize,
    /// A `submesh` came earlier: `cab` lines only count inside one, as Rigs of Rods
    /// reads them (a `cab` before any `submesh` is ignored there).
    in_submesh: bool,
    /// `cameras` lines, resolved after the file (they may precede `nodes`).
    cameras: Vec<(usize, Vec<String>)>,
    /// `fusedrag … autocalc`: its area coefficient.
    fuse_area_coef: f32,
    /// Legacy numbering: per `wheels` line its first generated number, count and
    /// wheel index.
    wheel_numbers: Vec<(u32, u32, u32)>,
    /// A `shocks2` line was discarded by validation: the next one is not validated
    /// (Rigs of Rods' behaviour, spec §4).
    shock2_skip_check: bool,
    /// Sections (`engine/docs/explicit-beams-sections-triggers-slidenodes-spec.md`
    /// §1): the module the parser is in (`None`: the root), the module names met
    /// (file order), and the module that spawns with the root.
    module: Option<String>,
    module_names: Vec<String>,
    selected: Option<String>,
    /// Per list, each item's module: true for the selected module's (the spawn
    /// order puts them after the root's, keyword by keyword).
    tags: Tags,
    /// Node lines met in the root and in the selected module, and whether a root
    /// node line followed a selected-module one (their indices then need a plan).
    root_nodes: u32,
    module_nodes: u32,
    root_after_module: bool,
    /// The node index plan of a second pass: the root's node count, and the root and
    /// module node lines met so far.
    node_plan: Option<(u32, u32, u32)>,
    /// Modules (root: "") with lines that generate nodes (legacy numbering), and the
    /// numbers unselected modules generated.
    wheel_modules: Vec<String>,
    unselected_numbers: Vec<(u32, u32)>,
    /// A line named a generated (wheel) node by its legacy number.
    generated_ref: bool,
    /// `set_inertia_defaults` with a positive delay is in force.
    inertia_defaults: bool,
}

/// Module tags of a [`TruckFile`]'s lists, item by item.
#[derive(Clone, Debug, Default)]
struct Tags {
    beams: Vec<bool>,
    wheels: Vec<bool>,
    triangles: Vec<bool>,
    cinecams: Vec<bool>,
    wings: Vec<bool>,
    slidenodes: Vec<bool>,
    tyre_beams: Vec<bool>,
}

impl Parser {
    fn warn(&mut self, msg: String) {
        if self.out.warnings.len() < 200 {
            self.out
                .warnings
                .push(format!("line {}: {msg}", self.line_no));
        }
    }

    fn error(&mut self, msg: String) {
        let m = format!("line {}: {msg}", self.line_no);
        self.out.errors.push(m);
    }

    /// The parser is inside a module that does not spawn: its lines are dropped
    /// (directives still apply).
    fn unselected(&self) -> bool {
        self.module.is_some() && self.module != self.selected
    }

    /// Tag every list item added since the last call with the current module.
    fn sync_tags(&mut self) {
        let m = self.module.is_some();
        let pad = |v: &mut Vec<bool>, n: usize| v.resize(n, m);
        pad(&mut self.tags.beams, self.out.beams.len());
        pad(&mut self.tags.wheels, self.out.wheels.len());
        pad(&mut self.tags.triangles, self.out.triangles.len());
        pad(&mut self.tags.cinecams, self.out.cinecams.len());
        pad(&mut self.tags.wings, self.out.wings.len());
        pad(&mut self.tags.slidenodes, self.out.slidenodes.len());
        pad(&mut self.tags.tyre_beams, self.out.tyre_beams.len());
    }

    /// `section <version> <name>` and `end_section`: switch module.
    fn switch_module(&mut self, name: Option<String>) {
        self.sync_tags();
        if let Some(n) = &name {
            if !self.module_names.contains(n) {
                self.module_names.push(n.clone());
            }
        }
        self.module = name;
    }

    /// A line of an unselected module: nothing spawns, but node-generating lines
    /// still take legacy numbers (spec §1.4).
    fn count_unselected(&mut self, t: &[&str]) {
        if !self.sequential() {
            return;
        }
        let per_ray = match self.section {
            Section::Nodes => {
                self.next_number += 1;
                return;
            }
            Section::Cinecam => {
                self.next_number += 1;
                return;
            }
            Section::Wheels | Section::MeshWheels | Section::MeshWheels2 => 2,
            Section::Wheels2 | Section::FlexBodyWheels => 4,
            _ => return,
        };
        let rays_col = match self.section {
            Section::Wheels => 2,
            _ => 3,
        };
        let rays = f(t, rays_col, 12.0).max(1.0) as u32;
        let module = self.module.clone().unwrap_or_default();
        if !self.wheel_modules.contains(&module) {
            self.wheel_modules.push(module);
        }
        self.unselected_numbers
            .push((self.next_number, per_ray * rays));
        self.next_number += per_ray * rays;
    }

    fn node(&mut self, tok: &str) -> Option<u32> {
        let mut r = self.ids.get(tok).copied();
        // Legacy numbering (fileformatversion < 450): RoR reads a node reference as an
        // integer and takes its absolute value (`_ParseNodeRef`), so `-65` is node 65.
        if r.is_none() && self.sequential() {
            if let Ok(n) = tok.parse::<i64>() {
                r = self.ids.get(&n.abs().to_string()).copied();
            }
        }
        if r.is_none() {
            self.warn(format!("unknown node `{tok}`"));
        }
        r
    }

    fn sequential(&self) -> bool {
        self.out.file_format_version < 450
    }

    /// Effective deform threshold of a new beam (RoR spawn rules).
    fn deform_threshold(&self) -> f32 {
        let mut d = self.defaults.deform;
        // `enable_advanced_deformation` counts as of the `set_beam_defaults` line.
        if self.defaults.user && !self.defaults.advanced && d < DEFAULT_DEFORM {
            d = DEFAULT_DEFORM;
        }
        if !self.defaults.plastic_set && d < 100_000.0 {
            d = 100_000.0;
        }
        d * self.scale.deform
    }

    fn beam(&self, a: u32, b: u32) -> SoftBeam {
        SoftBeam {
            a,
            b,
            kind: BeamKind::Normal,
            spring: self.defaults.spring * self.scale.spring,
            damp: self.defaults.damp * self.scale.damp,
            deform: self.deform_threshold(),
            strength: self.defaults.brk * self.scale.brk,
            plastic_coef: self.defaults.plastic,
            break_group: self.detacher_group,
            ..Default::default()
        }
    }

    fn keyword(&mut self, word: &str, t: &[&str]) -> bool {
        let section = match word {
            "globals" => Section::Globals,
            "nodes" => Section::Nodes,
            "nodes2" => Section::Nodes2,
            "beams" => Section::Beams,
            "shocks" => Section::Shocks,
            "shocks2" => Section::Shocks2,
            "shocks3" => Section::Shocks3,
            "cab" => Section::Cab,
            "contacters" => Section::Contacters,
            "wheels" => Section::Wheels,
            "wheels2" => Section::Wheels2,
            "meshwheels" => Section::MeshWheels,
            "meshwheels2" => Section::MeshWheels2,
            "flexbodywheels" => Section::FlexBodyWheels,
            "engine" => Section::Engine,
            "brakes" => Section::Brakes,
            "hydros" => Section::Hydros,
            "commands" => Section::Commands,
            "commands2" => Section::Commands2,
            "wings" => Section::Wings,
            "fusedrag" => Section::FuseDrag,
            "fixes" => Section::Fixes,
            "minimass" => Section::Minimass,
            "cameras" => Section::Cameras,
            "cinecam" => Section::Cinecam,
            "triggers" => Section::Triggers,
            "animators" => Section::Animators,
            "slidenodes" => Section::SlideNodes,
            "rubble_tyre_pose" => Section::TyrePose,
            "rubble_spawn_contacts" => Section::SpawnContacts,
            "rubble_contact_breaks" => Section::ContactBreaks,
            "rubble_jbeam_plastic" => Section::JbeamPlastic,
            "rubble_tension_yield" => Section::TensionYield,
            "rubble_damp_cutoff" => Section::DampCutoff,
            "rubble_aero_tri" => Section::AeroTri,
            // Known sections this importer does not simulate (phase 3 / cosmetic).
            "engoption"
            | "engturbo"
            | "torquecurve"
            | "axles"
            | "flexbodies"
            | "props"
            | "rotators"
            | "rotators2"
            | "ties"
            | "ropes"
            | "ropables"
            | "turboprops"
            | "turboprops2"
            | "turbojets"
            | "pistonprops"
            | "screwprops"
            | "soundsources"
            | "soundsources2"
            | "flares"
            | "flares2"
            | "flares3"
            | "exhausts"
            | "particles"
            | "materialflarebindings"
            | "managedmaterials"
            | "airbrakes"
            | "texcoords"
            | "railgroups"
            | "slidenode_connect_instantly"
            | "collisionboxes"
            | "camerarail"
            | "lockgroups"
            | "videocamera"
            | "interaxles"
            | "transfercase"
            | "nodecollision"
            | "description"
            | "help"
            | "comment"
            | "submesh_groundmodel_only"
            | "tractioncontrol"
            | "antilockbrakes"
            | "cruisecontrol"
            | "speedlimiter"
            | "guisettings"
            | "set_shadows"
            | "fileinfo"
            | "author"
            | "guid"
            | "extcamera"
            | "globeams" => Section::Skipped,
            _ => return false,
        };
        if section == Section::Cab && !self.in_submesh {
            self.warn("`cab` outside a `submesh`: ignored (Rigs of Rods ignores it)".into());
            self.section = Section::Skipped;
            return true;
        }
        if word == "railgroups" || word == "slidenode_connect_instantly" {
            self.error(format!(
                "`{word}` is not supported (slide-node rail groups and attachment changes)"
            ));
        }
        if section == Section::Skipped {
            *self.skipped.entry(word.to_string()).or_default() += 1;
            // Multi-line blocks that end with their own terminator.
            if word == "description" || word == "comment" {
                self.section = Section::Skipped;
            }
        }
        let _ = t;
        self.section = section;
        true
    }

    fn directive(&mut self, word: &str, t: &[&str]) -> bool {
        match word {
            "set_beam_defaults" => {
                // From the current values: an omitted field keeps its value, a
                // negative one takes the built-in default (spec §1.5).
                let d = BeamDefaults::default();
                let num = |i: usize| t.get(i).and_then(|s| s.parse::<f32>().ok());
                let field = |i: usize, current: f32, default: f32| match num(i) {
                    None => current,
                    Some(v) if v < 0.0 => default,
                    Some(v) => v,
                };
                let cur = self.defaults;
                self.defaults.spring = field(1, cur.spring, d.spring);
                self.defaults.damp = field(2, cur.damp, d.damp);
                self.defaults.deform = field(3, cur.deform, d.deform);
                self.defaults.brk = field(4, cur.brk, d.brk);
                if let Some(p) = num(7) {
                    if p >= 0.0 {
                        self.defaults.plastic = p;
                        self.defaults.plastic_set = true;
                    } else {
                        self.defaults.plastic = 0.0;
                        self.defaults.plastic_set = false;
                    }
                }
                self.defaults.user = true;
                self.defaults.advanced = self.out.advanced_deformation;
            }
            "set_beam_defaults_scale" => {
                // All four scales or the line is ignored; a fifth number is ignored.
                if (1..5).all(|i| t.get(i).is_some_and(|s| s.parse::<f32>().is_ok())) {
                    self.scale = Scale {
                        spring: fd(t, 1, 1.0),
                        damp: fd(t, 2, 1.0),
                        deform: fd(t, 3, 1.0),
                        brk: fd(t, 4, 1.0),
                    };
                } else {
                    self.warn("set_beam_defaults_scale needs four numbers: ignored".into());
                }
            }
            "set_inertia_defaults" => {
                let (start, stop) = (f(t, 1, 0.0), f(t, 2, 0.0));
                self.inertia_defaults = start > 0.0 || stop > 0.0;
            }
            "section" => {
                // `section <version> <name> …`: only the first name counts.
                if t.len() >= 3 {
                    let name = t[2].to_string();
                    if self.module.as_deref() != Some(&name) {
                        self.switch_module(Some(name));
                    }
                }
                self.section = Section::Skipped;
            }
            "end_section" => {
                if self.module.is_some() {
                    self.switch_module(None);
                }
                self.section = Section::Skipped;
            }
            // Ignored entirely, with the data lines after it (quirk).
            "sectionconfig" => self.section = Section::Skipped,
            "set_node_defaults" => {
                if t.len() < 2 {
                    self.warn("set_node_defaults needs the load weight: ignored".into());
                    return true;
                }
                let lw = f(t, 1, -1.0);
                let fr = f(t, 2, -1.0);
                self.node_defaults = NodeDefaults {
                    load_weight: (lw >= 0.0).then_some(lw),
                    friction: (fr >= 0.0).then_some(fr),
                };
            }
            "set_default_minimass" => self.out.minimass = fd(t, 1, DEFAULT_MINIMASS),
            "enable_advanced_deformation" => self.out.advanced_deformation = true,
            "rubble_full_contact" => self.out.full_contact = true,
            "rubble_exact_lengths" => self.out.exact_lengths = true,
            "rubble_beamng_profile" => self.out.beamng_profile = true,
            "rubble_drag_coef" => {
                let d = f(t, 1, 100.0);
                self.out.drag_coef = Some((d, f(t, 2, d)));
            }
            "rubble_rest_pose" => self.out.rest_ground = Some(f(t, 1, 0.0)),
            "detacher_group" => {
                self.detacher_group = if t.get(1) == Some(&"end") {
                    0
                } else {
                    f(t, 1, 0.0).abs() as u32
                };
            }
            "set_collision_range" => {
                self.out.collision_range = fd(t, 1, DEFAULT_COLLISION_RANGE);
                self.out.collision_range_set = true;
            }
            "submesh_groundmodel" => {
                self.out.submesh_groundmodel = t.get(1).map(|g| g.to_string());
            }
            "fileformatversion" => self.out.file_format_version = f(t, 1, 0.0) as u32,
            "forwardcommands"
            | "importcommands"
            | "rescuer"
            | "disabledefaultsounds"
            | "hideInChooser"
            | "lockgroup_default_nolock"
            | "rollon"
            | "slopebrake"
            | "submesh"
            | "backmesh"
            | "forset"
            | "set_skeleton_settings"
            | "set_managedmaterials_options"
            | "set_shadows"
            | "prop_camera_mode"
            | "flexbody_camera_mode"
            | "add_animation"
            | "speedlimiter"
            | "guid"
            | "author"
            | "fileinfo"
            | "enable_advanced_deformation_end"
            // `extcamera <mode> [node]`: one line, not a block (a camera setting).
            | "extcamera"
            | "end_description"
            | "end_comment" => {
                if word == "end_description" || word == "end_comment" {
                    self.section = Section::None;
                }
                if word == "submesh" {
                    // A new submesh: its `cab` lines follow.
                    self.section = Section::None;
                    self.in_submesh = true;
                }
            }
            _ => return false,
        }
        true
    }

    fn data(&mut self, t: &[&str]) {
        if self.unselected() {
            self.count_unselected(t);
            return;
        }
        match self.section {
            Section::None => {
                let w = t[0].to_string();
                self.warn(format!("unrecognised line `{w}`"));
            }
            Section::Skipped => {}
            Section::Globals => {
                self.out.dry_mass = f(t, 0, 0.0);
                self.out.cargo_mass = f(t, 1, 0.0);
            }
            Section::Minimass => self.out.minimass = fd(t, 0, DEFAULT_MINIMASS),
            Section::Nodes | Section::Nodes2 => self.parse_node(t),
            Section::Beams => {
                if t.len() < 2 {
                    return;
                }
                // A beam to a node a `wheels` line generated (legacy numbering):
                // kept apart, for the solvers that build those nodes.
                let ends = [self.tyre_ref(t[0]), self.tyre_ref(t[1])];
                let (a, b) = if ends.iter().any(Option::is_some) {
                    let file = |p: &mut Self, tok: &str, e: Option<TruckNodeRef>| match e {
                        Some(r) => Some(r),
                        None => p.node(tok).map(TruckNodeRef::File),
                    };
                    let (Some(ra), Some(rb)) =
                        (file(self, t[0], ends[0]), file(self, t[1], ends[1]))
                    else {
                        return;
                    };
                    let mut beam = self.beam(0, 0);
                    Self::beam_options(&mut beam, t);
                    self.out.tyre_beams.push(TruckTyreBeam {
                        position: self.out.beams.len(),
                        a: ra,
                        b: rb,
                        beam,
                    });
                    return;
                } else {
                    let (Some(a), Some(b)) = (self.node(t[0]), self.node(t[1])) else {
                        return;
                    };
                    (a, b)
                };
                let mut beam = self.beam(a, b);
                Self::beam_options(&mut beam, t);
                // (Rubble's extension: a fifth field, its rest length over
                // its spawn length.)
                if let Some(p) = t
                    .get(4)
                    .and_then(|p| p.parse::<f32>().ok())
                    .filter(|&p| p > 0.0)
                {
                    beam.precompression = p;
                }
                self.out.beams.push(beam);
            }
            Section::Shocks | Section::Shocks2 | Section::Shocks3 => self.parse_shock(t),
            Section::Hydros => {
                if t.len() < 3 {
                    return;
                }
                let (Some(a), Some(b)) = (self.node(t[0]), self.node(t[1])) else {
                    return;
                };
                let mut beam = self.beam(a, b);
                self.out
                    .actuator_gates
                    .push((self.out.beams.len(), beam.deform));
                beam.kind = BeamKind::Hydro;
                beam.hydro_factor = f(t, 2, 0.0);
                beam.deform = f32::INFINITY;
                if t.get(3).is_some_and(|o| o.contains('i')) {
                    beam.visible = false;
                }
                self.out.beams.push(beam);
            }
            _ => self.data_rest(t),
        }
    }

    /// A `beams` line's options (`i`, `r`, `s` with its break factor).
    fn beam_options(beam: &mut SoftBeam, t: &[&str]) {
        let opts = t.get(2).copied().unwrap_or("n");
        for c in opts.chars() {
            match c {
                'i' => beam.visible = false,
                'r' => beam.kind = BeamKind::Rope,
                's' => {
                    beam.kind = BeamKind::Support;
                    let factor = f(t, 3, DEFAULT_SUPPORT_BREAK_FACTOR);
                    beam.long_bound = if factor > 0.0 {
                        factor
                    } else {
                        DEFAULT_SUPPORT_BREAK_FACTOR
                    };
                }
                _ => {}
            }
        }
    }

    /// The node a legacy-numbered reference names among those a `wheels` line
    /// generated (its wheel and the node's place in its rings), if any.
    fn tyre_ref(&mut self, tok: &str) -> Option<TruckNodeRef> {
        if !self.sequential() || self.ids.contains_key(tok) {
            return None;
        }
        let n = tok.parse::<i64>().ok()?.unsigned_abs() as u32;
        let r = self
            .wheel_numbers
            .iter()
            .find_map(|&(first, count, wheel)| {
                (n >= first && n < first + count).then_some(TruckNodeRef::Tyre(wheel, n - first))
            });
        if r.is_some()
            || self
                .unselected_numbers
                .iter()
                .any(|&(first, count)| n >= first && n < first + count)
        {
            self.generated_ref = true;
        }
        r
    }

    fn data_rest(&mut self, t: &[&str]) {
        match self.section {
            Section::Commands | Section::Commands2 => {
                if t.len() < 2 {
                    return;
                }
                let (Some(a), Some(b)) = (self.node(t[0]), self.node(t[1])) else {
                    return;
                };
                // Key-bound actuators: fixed-length hydros for the solvers without
                // command input; the explicit solver reads `commands`.
                let mut beam = self.beam(a, b);
                let index = self.out.beams.len();
                self.out.actuator_gates.push((index, beam.deform));
                beam.kind = BeamKind::Hydro;
                beam.hydro_factor = 0.0;
                beam.deform = f32::INFINITY;
                self.out.beams.push(beam);
                // commands2: two rates; commands: one rate for both.
                let two = self.section == Section::Commands2;
                let r = |i: usize| f(t, i, 0.0);
                let (shorten, lengthen, base) = if two {
                    (r(2), r(3), 4)
                } else {
                    (r(2), r(2), 3)
                };
                let key = |i: usize| t.get(i).and_then(|s| s.parse::<i32>().ok()).unwrap_or(0);
                let opts = t.get(base + 4).copied().unwrap_or("n");
                // The first of o, p, c wins; the other two are dropped.
                let mode = opts.chars().find(|c| matches!(c, 'o' | 'p' | 'c'));
                // Start and stop delays after the description (absent: none).
                let delay = |i: usize| t.get(i).map(|s| s.parse::<f32>().ok());
                let quiet = |d: Option<Option<f32>>| match d {
                    None => true,
                    Some(v) => v == Some(0.0),
                };
                let no_inertia =
                    !self.inertia_defaults && quiet(delay(base + 6)) && quiet(delay(base + 7));
                self.out.commands.push(TruckCommand {
                    beam: index,
                    shorten_rate: shorten,
                    lengthen_rate: lengthen,
                    max_contraction: r(base),
                    max_extension: r(base + 1),
                    contract_key: key(base + 2),
                    extend_key: key(base + 3),
                    rope: opts.contains('r'),
                    centering: mode == Some('c'),
                    one_press: matches!(mode, Some('o' | 'p')),
                    no_inertia,
                });
            }
            Section::Triggers => self.parse_trigger(t),
            Section::Animators => self.parse_animator(t),
            Section::SlideNodes => self.parse_slidenode(t),
            Section::Wings => self.parse_wing(t),
            Section::FuseDrag => {
                if t.len() < 3 {
                    self.warn("fusedrag line needs front node, rear node and width".into());
                    return;
                }
                let (Some(front), Some(rear)) = (self.node(t[0]), self.node(t[1])) else {
                    return;
                };
                let (width, airfoil) = if t[2] == "autocalc" {
                    // The width is set after the file (it needs every node).
                    (f32::NAN, t.get(4))
                } else {
                    (f(t, 2, 0.0), t.get(3))
                };
                self.fuse_area_coef = if t[2] == "autocalc" {
                    f(t, 3, 1.0)
                } else {
                    1.0
                };
                self.out.fusedrag = Some(TruckFuseDrag {
                    front,
                    rear,
                    width,
                    airfoil: airfoil.copied().unwrap_or("NACA0009.afl").to_string(),
                });
            }
            Section::Cab => {
                if t.len() < 3 {
                    return;
                }
                let (Some(a), Some(b), Some(c)) =
                    (self.node(t[0]), self.node(t[1]), self.node(t[2]))
                else {
                    return;
                };
                let opts = t.get(3).copied().unwrap_or("");
                let solid = opts.chars().any(|c| matches!(c, 'c' | 'p' | 'u'));
                let buoyant = opts.chars().any(|c| matches!(c, 'D' | 'F' | 'S'));
                // (Rubble's extension, spec §18: a fifth field is its break
                // group, a sixth the edges it breaks with, bit k for nodes k
                // to k+1, all three if absent.)
                let group = t.get(4).and_then(|g| g.parse::<u32>().ok());
                let sides = t.get(5).and_then(|m| m.parse::<u8>().ok()).unwrap_or(7) & 7;
                let tri = SoftTriangle {
                    nodes: [a, b, c],
                    collision: solid || buoyant,
                    break_group: group.unwrap_or(0),
                    break_sides: if group.is_some() { sides } else { 0 },
                    ..Default::default()
                };
                // A line with a letter of each group is two collision triangles in
                // Rigs of Rods (adjacent, same nodes).
                if solid && buoyant {
                    self.out.triangles.push(tri.clone());
                }
                self.out.triangles.push(tri);
            }
            Section::SpawnContacts => {
                let n: Vec<u32> = t.iter().filter_map(|x| x.parse::<u32>().ok()).collect();
                if n.len() >= 4 {
                    self.out.spawn_contacts.push(([n[0], n[1], n[2]], n[3]));
                } else {
                    self.warn("rubble_spawn_contacts needs a, b, c, node: ignored".into());
                }
            }
            Section::ContactBreaks => {
                let g = t.first().and_then(|x| x.parse::<u32>().ok());
                let force = t.get(1).and_then(|x| x.parse::<f32>().ok());
                let nodes: Option<Vec<u32>> = t[2.min(t.len())..]
                    .iter()
                    .map(|x| x.parse::<u32>().ok())
                    .collect();
                match (g, force, nodes) {
                    (Some(g), Some(force), Some(nodes))
                        if g > 0 && force > 0.0 && !nodes.is_empty() =>
                    {
                        self.out.contact_breaks.push((g, force, nodes))
                    }
                    _ => self.warn(
                        "rubble_contact_breaks needs group (> 0), force_N (> 0), nodes…: ignored"
                            .into(),
                    ),
                }
            }
            Section::JbeamPlastic => {
                let n: Vec<f32> = t.iter().filter_map(|x| x.parse::<f32>().ok()).collect();
                if n.len() >= 5 && n[0] >= 0.0 && n[1] >= 0.0 {
                    self.out
                        .jbeam_plastic
                        .push((n[0] as u32, n[1] as u32, n[2], n[3], n[4]));
                } else {
                    self.warn(
                        "rubble_jbeam_plastic needs a, b, deformLimitStress, deformLimit, deformLimitExpansion: ignored"
                            .into(),
                    );
                }
            }
            Section::DampCutoff => {
                let n: Vec<f32> = t.iter().filter_map(|x| x.parse::<f32>().ok()).collect();
                if n.len() >= 3 && n[0] >= 0.0 && n[1] >= 0.0 && n[2] > 0.0 {
                    self.out.damp_cutoff.push((n[0] as u32, n[1] as u32, n[2]));
                } else {
                    self.warn("rubble_damp_cutoff needs a, b, dampCutoffHz (> 0): ignored".into());
                }
            }
            Section::AeroTri => {
                let n: Vec<f32> = t.iter().filter_map(|x| x.parse::<f32>().ok()).collect();
                if n.len() >= 5 && n[..3].iter().all(|&x| x >= 0.0) {
                    self.out
                        .aero_tri
                        .push(([n[0] as u32, n[1] as u32, n[2] as u32], n[3], n[4]));
                } else {
                    self.warn("rubble_aero_tri needs a, b, c, dragCoef, liftCoef: ignored".into());
                }
            }
            Section::TensionYield => {
                let n: Vec<f32> = t.iter().filter_map(|x| x.parse::<f32>().ok()).collect();
                if n.len() >= 3 && n[0] >= 0.0 && n[1] >= 0.0 && n[2] > 0.0 {
                    self.out
                        .tension_yield
                        .push((n[0] as u32, n[1] as u32, n[2]));
                } else {
                    self.warn("rubble_tension_yield needs a, b, Tt_N (> 0): ignored".into());
                }
            }
            Section::TyrePose => {
                if t.len() >= 7 {
                    self.out.tyre_rest_axle.push((
                        f(t, 0, 0.0) as u32,
                        [f(t, 1, 0.0), f(t, 2, 0.0), f(t, 3, 0.0)],
                        [f(t, 4, 0.0), f(t, 5, 0.0), f(t, 6, 0.0)],
                    ));
                } else if t.len() >= 5 {
                    self.out.tyre_pose.push((
                        f(t, 0, 0.0) as u32,
                        f(t, 1, 0.0) as u32,
                        [f(t, 2, 0.0), f(t, 3, 0.0), f(t, 4, 0.0)],
                    ));
                } else {
                    self.warn("rubble_tyre_pose needs wheel, node, x, y, z: ignored".into());
                }
            }
            Section::Contacters => {
                for tok in t {
                    if let Some(n) = self.node(tok) {
                        self.out.contacters.push(n);
                    }
                }
            }
            Section::Fixes => {
                for tok in t {
                    if let Some(n) = self.node(tok) {
                        self.out.nodes[n as usize].fixed = true;
                    }
                }
            }
            Section::Wheels
            | Section::Wheels2
            | Section::MeshWheels
            | Section::MeshWheels2
            | Section::FlexBodyWheels => self.parse_wheel(t),
            Section::Engine => {
                let mut ratios: Vec<f32> = t
                    .iter()
                    .skip(4)
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                if let Some(end) = ratios.iter().position(|&r| r < 0.0) {
                    ratios.truncate(end);
                }
                // ratios = [reverse, neutral, forward…]
                let reverse = ratios.first().copied().unwrap_or(3.0);
                let mut gears: Vec<f32> = ratios.iter().skip(2).copied().collect();
                // Rubble's gearbox holds 8 ratios with reverse: keep 7 forward spread
                // over the range (first and top included) and say so.
                if gears.len() > 7 {
                    let n = gears.len();
                    let picked: Vec<f32> = (0..7).map(|k| gears[(k * (n - 1) + 3) / 6]).collect();
                    self.warn(format!(
                        "{n} forward gears: Rubble's gearbox holds 7 with reverse, kept {picked:?}"
                    ));
                    gears = picked;
                }
                self.out.engine = Some(SoftEngine {
                    shift_down_rpm: f(t, 0, 1500.0),
                    shift_up_rpm: f(t, 1, 4500.0),
                    max_torque: f(t, 2, 400.0),
                    diff_ratio: f(t, 3, 3.5),
                    reverse_ratio: reverse,
                    gear_ratios: if gears.is_empty() {
                        SoftEngine::default().gear_ratios
                    } else {
                        gears
                    },
                    max_rpm: (f(t, 1, 4500.0) * 1.25).max(3000.0),
                });
            }
            Section::Brakes => self.out.brake_force = Some(f(t, 0, 30000.0)),
            Section::Cameras => {
                if t.len() >= 3 {
                    self.cameras
                        .push((self.line_no, t[..3].iter().map(|x| x.to_string()).collect()));
                }
            }
            Section::Cinecam => {
                // A cinecam is a generated node: it consumes a number.
                if self.sequential() {
                    self.next_number += 1;
                }
                if t.len() >= 11 {
                    let nodes: Option<Vec<u32>> = t[3..11].iter().map(|k| self.node(k)).collect();
                    if let Some(nodes) = nodes {
                        // ProcessCinecam: AddBeam with the beam defaults (unscaled break
                        // threshold), then the cinecam's spring and damping.
                        let mut beam = self.beam(0, 0);
                        beam.strength = self.defaults.brk;
                        beam.spring = f(t, 11, 8000.0);
                        beam.damp = f(t, 12, 800.0);
                        let m = f(t, 13, 20.0);
                        self.out.cinecams.push(TruckCinecam {
                            position: [f(t, 0, 0.0), f(t, 1, 0.0), f(t, 2, 0.0)],
                            nodes: nodes.try_into().unwrap(),
                            beam,
                            node_mass: if m > 0.0 { m } else { 20.0 },
                        });
                    }
                } else {
                    self.warn("cinecam line needs x, y, z and 8 nodes".into());
                }
            }
            // Handled in `data`.
            _ => {}
        }
    }

    fn parse_node(&mut self, t: &[&str]) {
        if t.len() < 4 {
            self.warn("node line needs id, x, y, z".into());
            return;
        }
        let id = t[0];
        if self.section == Section::Nodes && self.sequential() {
            if let Ok(n) = id.parse::<u32>() {
                if n != self.next_number {
                    self.warn(format!(
                        "node {n} out of sequence (expected {})",
                        self.next_number
                    ));
                }
            }
        }
        // Root nodes first, then the selected module's (spec §1.3): with a plan
        // (a root node line after a module one), each takes its planned index.
        let in_module = self.module.is_some();
        if in_module {
            self.module_nodes += 1;
        } else {
            self.root_nodes += 1;
            if self.module_nodes > 0 {
                self.root_after_module = true;
            }
        }
        let index = match &mut self.node_plan {
            Some((root, root_seen, module_seen)) => {
                if in_module {
                    *module_seen += 1;
                    *root + *module_seen - 1
                } else {
                    *root_seen += 1;
                    *root_seen - 1
                }
            }
            None => self.out.nodes.len() as u32,
        };
        let pos = [f(t, 1, 0.0), f(t, 2, 0.0), f(t, 3, 0.0)];
        let opts = t.get(4).copied().unwrap_or("n");
        let mut node = SoftNode {
            name: id.to_string(),
            position: pos,
            friction: self.node_defaults.friction.unwrap_or(1.0),
            ..Default::default()
        };
        let mut loaded = false;
        let mut weight = self.node_defaults.load_weight;
        for c in opts.chars() {
            match c {
                'l' => loaded = true,
                'c' => node.collision = false,
                _ => {}
            }
        }
        if let Some(w) = t.get(5).and_then(|s| s.parse::<f32>().ok()) {
            weight = Some(w);
        }
        if !loaded {
            weight = None;
        }
        self.ids.insert(id.to_string(), index);
        if self.node_plan.is_some() {
            self.out.nodes[index as usize] = node;
            self.out.loaded[index as usize] = (loaded, weight);
        } else {
            self.out.nodes.push(node);
            self.out.loaded.push((loaded, weight));
        }
        if self.section == Section::Nodes {
            self.next_number += 1;
        }
    }

    /// `n1, n2, contraction_limit, expansion_limit, short_action, long_action
    /// [, options [, boundary_timer]]` (spec §2.1–2.4).
    fn parse_trigger(&mut self, t: &[&str]) {
        if t.len() < 6 {
            self.warn("triggers line needs 6 fields: ignored".into());
            return;
        }
        let opts = t.get(6).copied().unwrap_or("");
        let mut refused = false;
        for c in opts.chars() {
            match c {
                'i' | 'c' => {}
                'x' | 'b' | 'B' | 'A' | 's' | 'h' | 'H' | 't' | 'E' => {
                    self.error(format!(
                        "trigger option `{c}` is not supported (only `c` and `i` are)"
                    ));
                    refused = true;
                }
                _ => self.warn(format!("trigger option `{c}` ignored")),
            }
        }
        let action = |i: usize| t[i].parse::<f32>().ok().map(|v| v as i64);
        let (short, long) = (action(4), action(5));
        if long == Some(-1) {
            self.error("trigger with long action -1 (a key blocker) is not supported".into());
            return;
        }
        if refused {
            return;
        }
        let Some(short) = short.filter(|k| (1..=COMMAND_KEYS as i64).contains(k)) else {
            self.warn(format!(
                "trigger short action `{}` outside 1…84: dropped",
                t[4]
            ));
            return;
        };
        let Some(long) = long.filter(|k| (0..=COMMAND_KEYS as i64).contains(k)) else {
            self.error(format!("trigger long action `{}` outside 0…84", t[5]));
            return;
        };
        let (Some(a), Some(b)) = (self.node(t[0]), self.node(t[1])) else {
            return;
        };
        let (sb0, lb0) = (f(t, 2, 0.0), f(t, 3, 0.0));
        // Option `c`: command-style bounds; lb is not made absolute (quirk).
        let (sb, lb) = if opts.contains('c') {
            ((sb0 - 1.0).abs(), lb0 - 1.0)
        } else {
            (sb0, lb0)
        };
        let mut beam = self.beam(a, b);
        let index = self.out.beams.len();
        self.out.actuator_gates.push((index, beam.deform));
        beam.kind = BeamKind::Hydro;
        beam.spring = 0.0;
        beam.damp = 0.0;
        // The unscaled break threshold.
        beam.strength = self.defaults.brk;
        beam.hydro_factor = 0.0;
        beam.deform = f32::INFINITY;
        beam.visible = false;
        self.out.beams.push(beam);
        self.out.triggers.push(TruckTrigger {
            beam: index,
            short_bound: sb,
            long_bound: lb,
            short_key: short as u32,
            long_key: long as u32,
        });
    }

    /// `n1, n2, factor, options` (spec §4): with no engine running, an actuator
    /// beam whose length never changes. Only `tacho` is accepted as a source.
    fn parse_animator(&mut self, t: &[&str]) {
        if t.len() < 4 {
            self.warn("animators line needs 4 fields: ignored".into());
            return;
        }
        for tok in &t[3..] {
            let known = matches!(*tok, "tacho" | "inv" | "vis" | "shortlimit" | "longlimit")
                || tok.parse::<f32>().is_ok();
            if !known {
                self.error(format!(
                    "animator source `{tok}` is not supported (only `tacho` with no engine)"
                ));
                return;
            }
        }
        let (Some(a), Some(b)) = (self.node(t[0]), self.node(t[1])) else {
            return;
        };
        let mut beam = self.beam(a, b);
        let index = self.out.beams.len();
        self.out.actuator_gates.push((index, beam.deform));
        beam.kind = BeamKind::Hydro;
        beam.hydro_factor = 0.0;
        beam.deform = f32::INFINITY;
        if t[3..].contains(&"inv") {
            beam.visible = false;
        }
        self.out.beams.push(beam);
        self.out.animators.push(index);
    }

    /// `node, rail nodes…, options` (spec §3.1).
    fn parse_slidenode(&mut self, t: &[&str]) {
        if t.len() < 2 {
            self.warn("slidenodes line needs a node and a rail: ignored".into());
            return;
        }
        let Some(node) = self.node(t[0]) else {
            return;
        };
        let mut s = TruckSlideNode {
            node,
            rail: Vec::new(),
            spring: 9_000_000.0,
            break_force: f32::INFINITY,
            tolerance: 0.0,
        };
        let mut options = false;
        for tok in &t[1..] {
            let mut chars = tok.chars();
            let letter = chars.next().map(|c| c.to_ascii_uppercase()).unwrap_or(' ');
            if matches!(letter, 'S' | 'B' | 'T' | 'R' | 'G' | 'D' | 'C') {
                options = true;
                let value = chars.as_str();
                match letter {
                    'S' | 'B' | 'T' => {
                        let Ok(v) = value.parse::<f32>() else {
                            self.error(format!(
                                "slidenodes token `{tok}` is not an option value (a rail node \
                                 named with an option letter is not supported)"
                            ));
                            return;
                        };
                        let v = v.abs();
                        match letter {
                            'S' => s.spring = v,
                            'B' => s.break_force = v,
                            _ => s.tolerance = v,
                        }
                    }
                    _ => {
                        self.error(format!(
                            "slidenodes option `{tok}` is not supported (attachment rate, \
                             distance, constraints and rail groups)"
                        ));
                        return;
                    }
                }
            } else if options {
                self.warn(format!("slidenodes option `{tok}` ignored"));
            } else {
                let range = tok
                    .split_once('-')
                    .and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)));
                match range {
                    Some((a, b)) => {
                        for i in a.min(b)..=a.max(b) {
                            let Some(n) = self.node(&i.to_string()) else {
                                return;
                            };
                            s.rail.push(n);
                        }
                    }
                    None => {
                        let Some(n) = self.node(tok) else {
                            return;
                        };
                        s.rail.push(n);
                    }
                }
            }
        }
        if s.rail.len() < 2 {
            self.error("slidenodes line with fewer than 2 rail nodes is not supported".into());
            return;
        }
        self.out.slidenodes.push(s);
    }

    fn parse_wing(&mut self, t: &[&str]) {
        if t.len() < 16 {
            self.warn("wings line needs 8 nodes and 8 texture coordinates".into());
            return;
        }
        let nodes: Option<Vec<u32>> = t[..8].iter().map(|k| self.node(k)).collect();
        let Some(nodes) = nodes else {
            return;
        };
        // A wing without an airfoil spawns nothing.
        let Some(airfoil) = t.get(20) else {
            self.warn("wing without an airfoil: ignored".into());
            return;
        };
        self.out.wings.push(TruckWing {
            nodes: nodes.try_into().unwrap(),
            control: t.get(16).and_then(|s| s.chars().next()).unwrap_or('n'),
            chord_point: f(t, 17, -1.0),
            min_deflection: f(t, 18, -1.0),
            max_deflection: f(t, 19, -1.0),
            airfoil: airfoil.to_string(),
            efficacy: f(t, 21, 1.0),
        });
    }

    fn parse_shock(&mut self, t: &[&str]) {
        let s = self.section;
        let need = match s {
            Section::Shocks => 7,
            _ => 13,
        };
        if t.len() < need {
            self.warn(format!("{s:?} line needs {need} fields"));
            return;
        }
        if s == Section::Shocks2 {
            // Any numeric field below -0.8 discards the line; the line after a
            // discarded one is not validated.
            if std::mem::take(&mut self.shock2_skip_check) {
            } else if (2..13).any(|i| f(t, i, 0.0) < -0.8) {
                self.warn("shocks2 line with a field below -0.8: discarded".into());
                self.shock2_skip_check = true;
                return;
            }
        }
        let (Some(a), Some(b)) = (self.node(t[0]), self.node(t[1])) else {
            return;
        };
        let (spring, damp, short, long, pre, opt_i) = match s {
            Section::Shocks => (
                f(t, 2, 0.0),
                f(t, 3, 0.0),
                f(t, 4, 0.0),
                f(t, 5, 0.0),
                f(t, 6, 1.0),
                7,
            ),
            // shocks2: spring/damp in and out — the "in" pair for the solvers
            // without the shocks2 force law ([`TruckShock2`] has the rest).
            Section::Shocks2 => (
                f(t, 2, 0.0),
                f(t, 3, 0.0),
                f(t, 10, 0.0),
                f(t, 11, 0.0),
                f(t, 12, 1.0),
                13,
            ),
            // shocks3: spring_in, damp_in, damp_in_fast, vsplit_in, spring_out, …
            _ => (
                f(t, 2, 0.0),
                f(t, 3, 0.0),
                f(t, 10, 0.0),
                f(t, 11, 0.0),
                f(t, 12, 1.0),
                13,
            ),
        };
        let mut beam = self.beam(a, b);
        if s == Section::Shocks2 {
            let opts = t.get(13).copied().unwrap_or("");
            for c in opts.chars() {
                if !matches!(c, 'i' | 's' | 'm' | 'M' | 'n' | 'v') {
                    self.warn(format!("shocks2 option `{c}` ignored"));
                }
            }
            self.out.shocks2.push(TruckShock2 {
                beam: self.out.beams.len(),
                spring_in: f(t, 2, 0.0),
                damp_in: f(t, 3, 0.0),
                prog_spring_in: f(t, 4, 0.0),
                prog_damp_in: f(t, 5, 0.0),
                spring_out: f(t, 6, 0.0),
                damp_out: f(t, 7, 0.0),
                prog_spring_out: f(t, 8, 0.0),
                prog_damp_out: f(t, 9, 0.0),
                short_bound: f(t, 10, 0.0),
                long_bound: f(t, 11, 0.0),
                precompression: f(t, 12, 0.0),
                metric: opts.contains('m'),
                abs_metric: opts.contains('M'),
                soft: opts.contains('s'),
                bump_spring: self.defaults.spring,
                bump_damp: self.defaults.damp,
                gate: beam.deform,
                strength: 4.0 * self.defaults.brk,
            });
        } else if s == Section::Shocks3 {
            self.out.shocks3.push(self.out.beams.len());
        }
        beam.kind = BeamKind::Bounded;
        beam.spring = spring;
        beam.damp = damp;
        // Past its bounds a RoR shock moves toward the beam defaults' spring and
        // damping (`shock_t::sbd_spring`, `sbd_damp`; unscaled).
        beam.deform = self.defaults.spring;
        beam.limit_damp = self.defaults.damp;
        // Shock strength: 4 × the unscaled break threshold.
        beam.strength = 4.0 * self.defaults.brk;
        beam.precompression = if pre > 0.0 { pre } else { 1.0 };
        let opts = t.get(opt_i).copied().unwrap_or("n");
        let metric = opts.contains('m');
        let l0 = self.rest(a, b) * beam.precompression;
        let (short, long) = if metric && l0 > 1e-6 {
            (short / l0, long / l0)
        } else {
            (short, long)
        };
        beam.short_bound = short.max(0.0);
        beam.long_bound = long.max(0.0);
        if opts.contains('i') {
            beam.visible = false;
        }
        self.out.beams.push(beam);
    }

    fn rest(&self, a: u32, b: u32) -> f32 {
        let pa = Vec3::from_array(self.out.nodes[a as usize].position);
        let pb = Vec3::from_array(self.out.nodes[b as usize].position);
        (pa - pb).length()
    }

    fn parse_wheel(&mut self, t: &[&str]) {
        let s = self.section;
        // Column of the first node (node1) and of the generated-node multiplier.
        let (kind, n1_col, nodes_per_ray): (&'static str, usize, u32) = match s {
            Section::Wheels => ("wheels", 3, 2),
            Section::Wheels2 => ("wheels2", 4, 4),
            Section::MeshWheels => ("meshwheels", 4, 2),
            Section::MeshWheels2 => ("meshwheels2", 4, 2),
            _ => ("flexbodywheels", 4, 4),
        };
        let mesh = matches!(s, Section::MeshWheels | Section::MeshWheels2);
        // meshwheels / meshwheels2 have 16 arguments, all required.
        let need = if mesh { 16 } else { n1_col + 7 };
        if t.len() < need {
            self.warn(format!("{kind} line too short"));
            return;
        }
        let rays = f(t, n1_col - 1, 12.0).max(1.0) as u32;
        let (Some(node1), Some(node2)) = (self.node(t[n1_col]), self.node(t[n1_col + 1])) else {
            return;
        };
        if mesh && node1 == node2 {
            self.warn(format!(
                "{kind} line with the same axle node twice: ignored"
            ));
            return;
        }
        let arm_tok = t[n1_col + 5];
        let arm = if arm_tok == "9999" || arm_tok == "-1" {
            None
        } else {
            self.ids.get(arm_tok).copied()
        };
        let radius = match s {
            Section::Wheels => f(t, 0, 0.4),
            // wheels2: rim radius, tyre radius; mesh/flexbody wheels: tyre radius first.
            Section::Wheels2 => f(t, 1, 0.4),
            _ => f(t, 0, 0.4),
        };
        let width = match s {
            Section::Wheels => f(t, 1, 0.2),
            _ => f(t, 2, 0.2),
        };
        // GetArgRigidityNode: 9999 is "none", anything else a node reference.
        let rigidity = if t[n1_col + 2] == "9999" {
            None
        } else {
            self.node(t[n1_col + 2])
        };
        self.out.wheels.push(TruckWheel {
            radius,
            width,
            rays,
            node1,
            node2,
            arm,
            braked: f(t, n1_col + 3, 1.0) as u32,
            propulsed: f(t, n1_col + 4, 0.0) as u32,
            mass: f(t, n1_col + 6, 20.0),
            kind,
            spring: f(t, n1_col + 7, 0.0),
            damp: f(t, n1_col + 8, 0.0),
            rim_spring: if s == Section::MeshWheels2 {
                self.defaults.spring
            } else {
                f(t, n1_col + 7, 0.0)
            },
            rim_damp: if s == Section::MeshWheels2 {
                self.defaults.damp
            } else {
                f(t, n1_col + 8, 0.0)
            },
            rigidity,
            beam: {
                // AddBeam: unscaled break threshold, scaled deform threshold.
                let mut b = self.beam(0, 0);
                b.strength = self.defaults.brk;
                b
            },
            node_friction: self.node_defaults.friction.unwrap_or(1.0),
        });
        if self.sequential() {
            let module = self.module.clone().unwrap_or_default();
            if !self.wheel_modules.contains(&module) {
                self.wheel_modules.push(module);
            }
            if matches!(kind, "wheels" | "meshwheels" | "meshwheels2") {
                let wheel = self.out.wheels.len() as u32 - 1;
                self.wheel_numbers
                    .push((self.next_number, nodes_per_ray * rays, wheel));
            }
            self.next_number += nodes_per_ray * rays;
        }
    }
}

/// Parse a `.truck`-family file into its file-level form, with its default section
/// configuration ([`parse_truck_config`]).
pub fn parse_truck(src: &str) -> TruckFile {
    parse_truck_config(src, None)
}

/// Parse a `.truck`-family file with the section configuration `config`
/// (`engine/docs/explicit-beams-sections-triggers-slidenodes-spec.md` §1): the root
/// module plus the `section` named `config`; `None` picks the first name in byte
/// order (Rigs of Rods' default), `Some("")` the root alone, and an unknown name the
/// root alone with a warning. [`TruckFile::section_configs`] lists the names and
/// [`TruckFile::section_config`] the one spawned. Each list holds the root's items
/// first, then the selected module's (the spawn order, keyword by keyword).
pub fn parse_truck_config(src: &str, config: Option<&str>) -> TruckFile {
    let first = parse_pass(src, None, None);
    if first.module_names.is_empty() {
        let mut out = first.out;
        if let Some(c) = config.filter(|c| !c.is_empty()) {
            out.warnings.push(format!(
                "section configuration `{c}` requested, the file has none"
            ));
        }
        return out;
    }
    let mut names = first.module_names.clone();
    names.sort();
    let chosen = match config {
        None => Some(names[0].clone()),
        Some("") => None,
        Some(c) if names.iter().any(|n| n == c) => Some(c.to_string()),
        Some(c) => {
            let _ = c;
            None
        }
    };
    let mut p = parse_pass(src, chosen.clone(), None);
    if p.root_after_module {
        // The selected module's nodes follow every root node.
        let plan = (p.root_nodes, p.root_nodes + p.module_nodes);
        p = parse_pass(src, chosen.clone(), Some(plan));
    }
    if let Some(c) = config.filter(|c| !c.is_empty() && chosen.is_none()) {
        p.out.warnings.push(format!(
            "section configuration `{c}` not found (have {}): the root alone spawns",
            names.join(", ")
        ));
    }
    if p.generated_ref && p.wheel_modules.len() > 1 {
        p.out.errors.push(
            "a line names a generated wheel node by its legacy number while more than one \
             section generates nodes: not supported"
                .into(),
        );
    }
    let tags = std::mem::take(&mut p.tags);
    let mut out = p.out;
    out.section_configs = names;
    out.section_config = chosen;
    interleave(&mut out, &tags);
    out
}

/// Stable-partition `v` by `tags` (false first); the old-to-new index map.
fn partition<T>(v: &mut Vec<T>, tags: &[bool]) -> Vec<usize> {
    let n = v.len();
    let mut order: Vec<usize> = (0..n)
        .filter(|&i| !tags.get(i).copied().unwrap_or(false))
        .collect();
    order.extend((0..n).filter(|&i| tags.get(i).copied().unwrap_or(false)));
    let mut map = vec![0usize; n];
    for (new, &old) in order.iter().enumerate() {
        map[old] = new;
    }
    let mut items: Vec<Option<T>> = std::mem::take(v).into_iter().map(Some).collect();
    *v = order.iter().map(|&i| items[i].take().unwrap()).collect();
    map
}

/// Put every list's root items before the selected module's, remapping the
/// references between lists (spec §1.3).
fn interleave(out: &mut TruckFile, tags: &Tags) {
    if !tags
        .beams
        .iter()
        .chain(&tags.wheels)
        .chain(&tags.triangles)
        .chain(&tags.cinecams)
        .chain(&tags.wings)
        .chain(&tags.slidenodes)
        .chain(&tags.tyre_beams)
        .any(|&t| t)
    {
        return;
    }
    // Tyre beams: their place among the file beams, in the new order.
    let root_beams = tags.beams.iter().filter(|&&t| !t).count();
    let positions: Vec<usize> = out
        .tyre_beams
        .iter()
        .enumerate()
        .map(|(k, tb)| {
            let module = tags.tyre_beams.get(k).copied().unwrap_or(false);
            let before = (0..tb.position.min(tags.beams.len()))
                .filter(|&i| tags.beams[i] == module)
                .count();
            if module {
                root_beams + before
            } else {
                before
            }
        })
        .collect();
    for (tb, p) in out.tyre_beams.iter_mut().zip(positions) {
        tb.position = p;
    }
    partition(&mut out.tyre_beams, &tags.tyre_beams);
    let beam = partition(&mut out.beams, &tags.beams);
    for c in &mut out.commands {
        c.beam = beam[c.beam];
    }
    for s in &mut out.shocks2 {
        s.beam = beam[s.beam];
    }
    for s in &mut out.shocks3 {
        *s = beam[*s];
    }
    for g in &mut out.actuator_gates {
        g.0 = beam[g.0];
    }
    for t in &mut out.triggers {
        t.beam = beam[t.beam];
    }
    for a in &mut out.animators {
        *a = beam[*a];
    }
    let wheel = partition(&mut out.wheels, &tags.wheels);
    let remap = |r: &mut TruckNodeRef| {
        if let TruckNodeRef::Tyre(w, k) = *r {
            *r = TruckNodeRef::Tyre(wheel[w as usize] as u32, k);
        }
    };
    for tb in &mut out.tyre_beams {
        remap(&mut tb.a);
        remap(&mut tb.b);
    }
    partition(&mut out.triangles, &tags.triangles);
    partition(&mut out.cinecams, &tags.cinecams);
    partition(&mut out.wings, &tags.wings);
    partition(&mut out.slidenodes, &tags.slidenodes);
}

/// One pass over the file with `selected` as the spawning module; `node_plan`
/// (the root's and the total node count) places the module's nodes after the
/// root's.
fn parse_pass(src: &str, selected: Option<String>, node_plan: Option<(u32, u32)>) -> Parser {
    let mut p = Parser {
        out: TruckFile {
            minimass: DEFAULT_MINIMASS,
            collision_range: DEFAULT_COLLISION_RANGE,
            ..Default::default()
        },
        section: Section::None,
        defaults: BeamDefaults::default(),
        scale: Scale::default(),
        node_defaults: NodeDefaults::default(),
        detacher_group: 0,
        ids: HashMap::new(),
        next_number: 0,
        skipped: HashMap::new(),
        line_no: 0,
        in_submesh: false,
        cameras: Vec::new(),
        fuse_area_coef: 1.0,
        wheel_numbers: Vec::new(),
        shock2_skip_check: false,
        module: None,
        module_names: Vec::new(),
        selected,
        tags: Tags::default(),
        root_nodes: 0,
        module_nodes: 0,
        root_after_module: false,
        node_plan: None,
        wheel_modules: Vec::new(),
        unselected_numbers: Vec::new(),
        generated_ref: false,
        inertia_defaults: false,
    };
    if let Some((root, total)) = node_plan {
        p.node_plan = Some((root, 0, 0));
        p.out.nodes = vec![SoftNode::default(); total as usize];
        p.out.loaded = vec![(false, None); total as usize];
    }
    let mut have_title = false;
    for (i, raw) in src.lines().enumerate() {
        p.line_no = i + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with("//") {
            continue;
        }
        if !have_title {
            p.out.title = line.to_string();
            have_title = true;
            continue;
        }
        let t = tokens(line);
        if t.is_empty() {
            continue;
        }
        let word = t[0];
        if word == "end" {
            break;
        }
        if p.section == Section::Skipped && (word == "end_description" || word == "end_comment") {
            p.section = Section::None;
            continue;
        }
        if word == "minimass" && t.len() > 1 {
            // Not Rigs of Rods syntax: `minimass` is a block keyword (value on the next
            // line); RoR reads this line as data of the section before it.
            p.warn(format!(
                "`{}`: minimass takes its value on the next line; line ignored",
                t.join(" ")
            ));
            continue;
        }
        if t.len() == 1
            && word.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && p.keyword(word, &t)
        {
            continue;
        }
        if p.directive(word, &t) {
            continue;
        }
        if word.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && !matches!(
                p.section,
                Section::Nodes2
                    | Section::Nodes
                    | Section::Beams
                    | Section::Cab
                    | Section::Contacters
                    | Section::Fixes
                    | Section::Skipped
                    | Section::Shocks
                    | Section::Shocks2
                    | Section::Shocks3
                    | Section::Hydros
                    | Section::Commands
                    | Section::Commands2
                    | Section::Wings
                    | Section::FuseDrag
                    | Section::Cameras
                    | Section::Triggers
                    | Section::Animators
                    | Section::SlideNodes
            )
        {
            // An unknown keyword: skip its block with a warning.
            if t.len() == 1 {
                *p.skipped.entry(word.to_string()).or_default() += 1;
                p.section = Section::Skipped;
                continue;
            }
        }
        p.data(&t);
    }
    for (line, t) in std::mem::take(&mut p.cameras) {
        p.line_no = line;
        let (Some(center), Some(back), Some(left)) = (p.node(&t[0]), p.node(&t[1]), p.node(&t[2]))
        else {
            continue;
        };
        if p.out.cameras.is_none() {
            p.out.cameras = Some(RefNodes { center, back, left });
        }
    }
    // `fusedrag … autocalc`: z extent × y extent of the file's nodes (as written)
    // × the area coefficient, the extents starting at 1000 / −1000.
    let coef = p.fuse_area_coef;
    if let Some(fd) = p.out.fusedrag.as_mut().filter(|fd| fd.width.is_nan()) {
        let (mut lo, mut hi) = ([1000.0f32; 2], [-1000.0f32; 2]);
        for n in &p.out.nodes {
            for (k, v) in [n.position[2], n.position[1]].into_iter().enumerate() {
                lo[k] = lo[k].min(v);
                hi[k] = hi[k].max(v);
            }
        }
        fd.width = (hi[0] - lo[0]) * (hi[1] - lo[1]) * coef;
    }
    let mut skipped: Vec<_> = std::mem::take(&mut p.skipped).into_iter().collect();
    skipped.sort();
    for (s, _) in skipped {
        p.out
            .warnings
            .push(format!("section `{s}` skipped (not simulated)"));
    }
    for &c in &p.out.contacters.clone() {
        p.out.nodes[c as usize].contacter = true;
    }
    p.sync_tags();
    p
}

/// RoR's node mass distribution (computed once after spawn):
/// cargo on loaded nodes, dry mass spread by beam length, minimass floor.
/// `wheels` add their spokes' length to the axle nodes (tyre nodes are not created;
/// their beams still distribute dry mass to the axle).
pub fn distribute_masses(file: &mut TruckFile) {
    let n = file.nodes.len();
    let mut mass = vec![0.0f32; n];
    // 1. Cargo on loaded nodes.
    let shared: Vec<usize> = (0..n)
        .filter(|&i| file.loaded[i].0 && file.loaded[i].1.is_none())
        .collect();
    for i in 0..n {
        if let (true, Some(w)) = file.loaded[i] {
            mass[i] = w;
        }
    }
    if !shared.is_empty() {
        let each = file.cargo_mass / shared.len() as f32;
        for &i in &shared {
            mass[i] = each;
        }
    }
    // 2. Dry mass by beam length: every beam end at a (non-tyre) node holds L/2.
    let pos = |i: u32| Vec3::from_array(file.nodes[i as usize].position);
    let mut halves: Vec<(u32, f32)> = Vec::new();
    for b in &file.beams {
        let l = (pos(b.a) - pos(b.b)).length()
            * if b.kind == BeamKind::Bounded {
                b.precompression
            } else {
                1.0
            };
        halves.push((b.a, 0.5 * l));
        halves.push((b.b, 0.5 * l));
    }
    // Wheel spokes from each axle node to the tyre rings, as RoR builds them
    // (`BuildWheelBeams`): per ray, one to its own ring at the radius and one across
    // the axle (RoR spans the axle's length, not the declared width). Only the axle
    // end holds mass; tyre nodes are excluded, and rim beams join tyre nodes only.
    for w in &file.wheels {
        let axle = (pos(w.node2) - pos(w.node1)).length();
        let near = w.radius;
        let across = (w.radius * w.radius + axle * axle).sqrt();
        for node in [w.node1, w.node2] {
            halves.push((node, 0.5 * w.rays as f32 * (near + across)));
        }
    }
    // Cinecam beams: RoR's cinecam node and its 8 beams to the body. Both halves
    // count in the split; the cinecam's own share is lost (RoR then sets that node's
    // mass from the cinecam definition), the body ends keep theirs.
    let mut lost = 0.0;
    for c in &file.cinecams {
        for &k in &c.nodes {
            let l = (Vec3::from_array(c.position) - pos(k)).length();
            halves.push((k, 0.5 * l));
            lost += 0.5 * l;
        }
    }
    let total: f32 = halves.iter().map(|h| h.1).sum::<f32>() + lost;
    if total > 0.0 && file.dry_mass > 0.0 {
        for &(i, h) in &halves {
            mass[i as usize] += file.dry_mass * h / total;
        }
    }
    // 4. Minimass floor (loaded nodes exempt with `minimass … l`).
    for i in 0..n {
        if file.minimass_skip_loaded && file.loaded[i].0 {
            continue;
        }
        mass[i] = mass[i].max(file.minimass);
    }
    for (node, m) in file.nodes.iter_mut().zip(mass) {
        node.mass = m;
    }
}

/// Parse, distribute masses and lower a `.truck` file into a [`SoftVehicleDesc`].
pub fn load_truck(src: &str) -> SoftVehicleDesc {
    load_truck_config(src, None)
}

/// [`load_truck`] with a section configuration ([`parse_truck_config`]).
pub fn load_truck_config(src: &str, config: Option<&str>) -> SoftVehicleDesc {
    let mut file = parse_truck_config(src, config);
    distribute_masses(&mut file);
    lower(file)
}

fn lower(mut file: TruckFile) -> SoftVehicleDesc {
    // Cinecams: their node after the truck's nodes, their beams before the truck's
    // (RoR spawns cinecams before beams), mass the cinecam's raised to the minimum.
    let mut cine_beams = Vec::new();
    for c in &file.cinecams {
        let node = file.nodes.len() as u32;
        file.nodes.push(SoftNode {
            name: format!("cinecam{}", cine_beams.len() / 8),
            position: c.position,
            mass: c.node_mass.max(file.minimass),
            collision: false,
            contacter: false,
            self_collision: false,
            ..Default::default()
        });
        file.loaded.push((false, None));
        for &k in &c.nodes {
            cine_beams.push(SoftBeam {
                a: node,
                b: k,
                visible: false,
                ..c.beam.clone()
            });
        }
    }
    if !cine_beams.is_empty() {
        cine_beams.append(&mut file.beams);
        file.beams = cine_beams;
    }
    let mut desc = SoftVehicleDesc {
        name: file.title.clone(),
        nodes: file.nodes,
        beams: file.beams,
        triangles: file.triangles,
        engine: file.engine,
        brake_torque: file.brake_force,
        refs: file.cameras,
        collision_range: file.collision_range,
        warnings: file.warnings,
        ..Default::default()
    };
    // Nodes collide with the world unless flagged `c`; with collision cabs present,
    // the contacters are the nodes that hit other vehicles' triangles.
    let any_contacters = desc.nodes.iter().any(|n| n.contacter);
    if !any_contacters {
        for n in &mut desc.nodes {
            n.contacter = n.collision;
        }
    }
    for w in &file.wheels {
        // Steering: a hydro on the wheel's axle or arm nodes, or one beam from them, steers it.
        // The max angle is the hydro's travel over its lever about the axle midpoint.
        let mut steerable = false;
        let mut max_steer = 0.0f32;
        let p = |i: u32| Vec3::from_array(desc.nodes[i as usize].position);
        let hub = 0.5 * (p(w.node1) + p(w.node2));
        // The wheel's nodes and their beam neighbours (steering linkages are often one
        // beam away from the axle, as on the Agora bus).
        let mut ends: Vec<u32> = [Some(w.node1), Some(w.node2), w.arm]
            .into_iter()
            .flatten()
            .collect();
        let direct = ends.clone();
        for b in desc.beams.iter().filter(|b| b.kind != BeamKind::Hydro) {
            if direct.contains(&b.a) && !ends.contains(&b.b) {
                ends.push(b.b);
            } else if direct.contains(&b.b) && !ends.contains(&b.a) {
                ends.push(b.a);
            }
        }
        for b in desc
            .beams
            .iter()
            .filter(|b| b.kind == BeamKind::Hydro && b.hydro_factor != 0.0)
        {
            let Some(&end) = ends.iter().find(|&&e| b.a == e || b.b == e) else {
                continue;
            };
            let lever = (p(end) - hub).length().max(0.2);
            let travel = (p(b.a) - p(b.b)).length() * b.hydro_factor.abs();
            steerable = true;
            max_steer = max_steer.max(
                (travel / lever)
                    .atan()
                    .clamp(10f32.to_radians(), 45f32.to_radians()),
            );
        }
        desc.wheels.push(SoftWheel {
            node1: w.node1,
            node2: w.node2,
            arm: w.arm,
            radius: w.radius,
            width: w.width,
            mass: w.mass,
            braked: w.braked != 0,
            propulsion: match w.propulsed {
                1 => 1,
                2 => -1,
                _ => 0,
            },
            steerable,
            max_steer,
        });
    }
    desc
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "tiny test truck
; comment
globals
1000, 0, tracks/semi

set_beam_defaults -1, -1, 200000, 800000
nodes
0, 0, 0, 0
1, 1, 0, 0
2, 0, 1, 0, l, 75
3, 0, 0, 1

beams
0, 1
0, 2
0, 3
1, 2, i
1, 3, s
2, 3, r

submesh
cab
0, 1, 2, c
0, 2, 3
end
";

    #[test]
    fn parses_sections_and_defaults() {
        let f = parse_truck(TINY);
        assert_eq!(f.title, "tiny test truck");
        assert_eq!(f.nodes.len(), 4);
        assert_eq!(f.beams.len(), 6);
        assert_eq!(f.dry_mass, 1000.0);
        // deform below 400 kN is raised without enable_advanced_deformation.
        assert_eq!(f.beams[0].deform, DEFAULT_DEFORM);
        assert_eq!(f.beams[0].strength, 800_000.0);
        assert_eq!(f.beams[0].spring, DEFAULT_SPRING);
        assert!(!f.beams[3].visible);
        assert_eq!(f.beams[4].kind, BeamKind::Support);
        assert_eq!(f.beams[4].long_bound, DEFAULT_SUPPORT_BREAK_FACTOR);
        assert_eq!(f.beams[5].kind, BeamKind::Rope);
        assert_eq!(f.triangles.len(), 2);
        assert!(f.triangles[0].collision);
        assert!(!f.triangles[1].collision);
        assert_eq!(f.loaded[2], (true, Some(75.0)));
    }

    #[test]
    fn mass_rules_match_ror() {
        let mut f = parse_truck(TINY);
        distribute_masses(&mut f);
        // Loaded node keeps its override plus its dry share; the rest get dry mass
        // by beam length then the 50 kg floor.
        let total_dry: f32 = 1000.0;
        let l = [1.0f32, 1.0, 1.0, 2f32.sqrt(), 2f32.sqrt(), 2f32.sqrt()];
        let total: f32 = l.iter().sum::<f32>();
        // node 0 touches beams 0,1,2 (lengths 1,1,1): share = 1.5 / total.
        let expect0 = total_dry * 1.5 / total;
        assert!(
            (f.nodes[0].mass - expect0.max(50.0)).abs() < 1e-3,
            "{}",
            f.nodes[0].mass
        );
        assert!(f.nodes[2].mass > 75.0);
        let sum: f32 = f.nodes.iter().map(|n| n.mass).sum();
        assert!(
            sum >= total_dry + 75.0 - 1e-2,
            "dry + cargo at least, got {sum}"
        );
    }

    #[test]
    fn advanced_deformation_keeps_low_thresholds() {
        let src = TINY.replace(
            "set_beam_defaults -1, -1, 200000, 800000",
            "enable_advanced_deformation\nset_beam_defaults -1, -1, 200000, 800000, -1, -1, 0.5",
        );
        let f = parse_truck(&src);
        assert_eq!(f.beams[0].deform, 200_000.0);
        assert_eq!(f.beams[0].plastic_coef, 0.5);
    }

    #[test]
    fn cameras_before_nodes_and_axle_hydro_steering() {
        let src = "cam
cameras
0, 1, 2
nodes
0, 0, 0.5, 0
1, -1, 0.5, 0
2, 0, 0.5, 1
3, 0, 0.3, 1
4, 0, 0.3, 0.6
beams
0, 1
0, 2
2, 3
3, 4
hydros
3, 1, 0.2
wheels
0.3, 0.2, 12, 3, 4, 9999, 1, 1, 9999, 20, 1000, 10
end
";
        let d = load_truck(src);
        assert!(
            d.warnings.iter().all(|w| !w.contains("unknown node")),
            "{:?}",
            d.warnings
        );
        assert!(d.refs.is_some());
        assert!(
            d.wheels[0].steerable,
            "hydro on the axle node steers the wheel"
        );
    }

    #[test]
    fn sequential_numbering_counts_wheel_nodes() {
        let src = "seq
nodes
0, 0, 0, 0
1, 0, 0, 1
2, 0, 1, 0
wheels
0.5, 0.2, 6, 0, 1, 9999, 1, 1, 2, 20, 1000, 10
nodes
15, 1, 1, 1
beams
0, 15
end
";
        let f = parse_truck(src);
        // 3 nodes + 2×6 tyre nodes ⇒ the next file node is number 15, no warnings.
        assert!(
            f.warnings.iter().all(|w| !w.contains("out of sequence")),
            "{:?}",
            f.warnings
        );
        assert_eq!(f.wheels.len(), 1);
        assert_eq!(f.wheels[0].rays, 6);
        assert_eq!(f.beams.len(), 1);
    }
}
