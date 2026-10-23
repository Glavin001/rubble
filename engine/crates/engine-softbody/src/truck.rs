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
    Fixes,
    Minimass,
    Cameras,
    Cinecam,
    Skipped,
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
    pub nodes: Vec<SoftNode>,
    /// Per node: loaded flag and optional load-weight override.
    pub loaded: Vec<(bool, Option<f32>)>,
    pub beams: Vec<SoftBeam>,
    pub triangles: Vec<SoftTriangle>,
    pub contacters: Vec<u32>,
    pub wheels: Vec<TruckWheel>,
    pub engine: Option<SoftEngine>,
    pub brake_force: Option<f32>,
    pub cameras: Option<RefNodes>,
    pub warnings: Vec<String>,
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
    /// `cameras` lines, resolved after the file (they may precede `nodes`).
    cameras: Vec<(usize, Vec<String>)>,
}

impl Parser {
    fn warn(&mut self, msg: String) {
        if self.out.warnings.len() < 200 {
            self.out
                .warnings
                .push(format!("line {}: {msg}", self.line_no));
        }
    }

    fn node(&mut self, tok: &str) -> Option<u32> {
        let r = self.ids.get(tok).copied();
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
        if !self.out.advanced_deformation && d < DEFAULT_DEFORM {
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
            "fixes" => Section::Fixes,
            "minimass" => Section::Minimass,
            "cameras" => Section::Cameras,
            "cinecam" => Section::Cinecam,
            // Known sections this importer does not simulate (phase 3 / cosmetic).
            "engoption"
            | "engturbo"
            | "torquecurve"
            | "axles"
            | "flexbodies"
            | "props"
            | "slidenodes"
            | "rotators"
            | "rotators2"
            | "ties"
            | "ropes"
            | "ropables"
            | "triggers"
            | "wings"
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
            | "animators"
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
            | "fusedrag"
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
                let d = BeamDefaults::default();
                self.defaults.spring = fd(t, 1, d.spring);
                self.defaults.damp = fd(t, 2, d.damp);
                self.defaults.deform = fd(t, 3, d.deform);
                self.defaults.brk = fd(t, 4, d.brk);
                if t.len() > 7 {
                    self.defaults.plastic = fd(t, 7, 0.0);
                    self.defaults.plastic_set = true;
                } else {
                    self.defaults.plastic = 0.0;
                    self.defaults.plastic_set = false;
                }
            }
            "set_beam_defaults_scale" => {
                self.scale = Scale {
                    spring: fd(t, 1, 1.0),
                    damp: fd(t, 2, 1.0),
                    deform: fd(t, 3, 1.0),
                    brk: fd(t, 4, 1.0),
                };
            }
            "set_node_defaults" => {
                let lw = f(t, 1, -1.0);
                let fr = f(t, 2, -1.0);
                self.node_defaults = NodeDefaults {
                    load_weight: (lw >= 0.0).then_some(lw),
                    friction: (fr >= 0.0).then_some(fr),
                };
            }
            "set_default_minimass" => self.out.minimass = fd(t, 1, DEFAULT_MINIMASS),
            "enable_advanced_deformation" => self.out.advanced_deformation = true,
            "detacher_group" => {
                self.detacher_group = if t.get(1) == Some(&"end") {
                    0
                } else {
                    f(t, 1, 0.0).abs() as u32
                };
            }
            "set_collision_range" => self.out.collision_range = fd(t, 1, DEFAULT_COLLISION_RANGE),
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
            | "set_inertia_defaults"
            | "set_shadows"
            | "prop_camera_mode"
            | "flexbody_camera_mode"
            | "add_animation"
            | "sectionconfig"
            | "section"
            | "end_section"
            | "submesh_groundmodel"
            | "speedlimiter"
            | "guid"
            | "author"
            | "fileinfo"
            | "enable_advanced_deformation_end"
            | "end_description"
            | "end_comment" => {
                if word == "end_description" || word == "end_comment" {
                    self.section = Section::None;
                }
                if word == "submesh" {
                    // A new submesh: its `cab` lines follow.
                    self.section = Section::None;
                }
            }
            _ => return false,
        }
        true
    }

    fn data(&mut self, t: &[&str]) {
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
                let (Some(a), Some(b)) = (self.node(t[0]), self.node(t[1])) else {
                    return;
                };
                let mut beam = self.beam(a, b);
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
                beam.kind = BeamKind::Hydro;
                beam.hydro_factor = f(t, 2, 0.0);
                beam.deform = f32::INFINITY;
                if t.get(3).is_some_and(|o| o.contains('i')) {
                    beam.visible = false;
                }
                self.out.beams.push(beam);
            }
            Section::Commands | Section::Commands2 => {
                if t.len() < 2 {
                    return;
                }
                let (Some(a), Some(b)) = (self.node(t[0]), self.node(t[1])) else {
                    return;
                };
                // Key-bound actuators: simulated as fixed-length hydros (no input bound yet).
                let mut beam = self.beam(a, b);
                beam.kind = BeamKind::Hydro;
                beam.hydro_factor = 0.0;
                beam.deform = f32::INFINITY;
                self.out.beams.push(beam);
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
                let collision = opts
                    .chars()
                    .any(|c| matches!(c, 'c' | 'p' | 'u' | 'D' | 'F' | 'S'));
                self.out.triangles.push(SoftTriangle {
                    nodes: [a, b, c],
                    collision,
                });
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
                    self.warn(format!("{n} forward gears: Rubble's gearbox holds 7 with reverse, kept {picked:?}"));
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
                    self.cameras.push((self.line_no, t[..3].iter().map(|x| x.to_string()).collect()));
                }
            }
            Section::Cinecam => {
                // A cinecam is a generated node: it consumes a number.
                if self.sequential() {
                    self.next_number += 1;
                }
            }
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
        let index = self.out.nodes.len() as u32;
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
        self.out.nodes.push(node);
        self.out.loaded.push((loaded, weight));
        if self.section == Section::Nodes {
            self.next_number += 1;
        }
    }

    fn parse_shock(&mut self, t: &[&str]) {
        let s = self.section;
        let need = match s {
            Section::Shocks => 7,
            Section::Shocks2 => 11,
            _ => 13,
        };
        if t.len() < need {
            self.warn(format!("{s:?} line needs {need} fields"));
            return;
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
            // shocks2: spring/damp in and out — use the "in" pair for the spring-damper.
            Section::Shocks2 => (
                f(t, 2, 0.0),
                f(t, 3, 0.0),
                f(t, 8, 0.0),
                f(t, 9, 0.0),
                f(t, 10, 1.0),
                11,
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
        beam.kind = BeamKind::Bounded;
        beam.spring = spring;
        beam.damp = damp;
        beam.deform = f32::INFINITY;
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
        if t.len() < n1_col + 7 {
            self.warn(format!("{kind} line too short"));
            return;
        }
        let rays = f(t, n1_col - 1, 12.0).max(1.0) as u32;
        let (Some(node1), Some(node2)) = (self.node(t[n1_col]), self.node(t[n1_col + 1])) else {
            return;
        };
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
        });
        if self.sequential() {
            self.next_number += nodes_per_ray * rays;
        }
    }
}

/// Parse a `.truck`-family file into its file-level form.
pub fn parse_truck(src: &str) -> TruckFile {
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
        cameras: Vec::new(),
    };
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
            // Inline form: `minimass 50 l`.
            p.out.minimass = fd(&t, 1, DEFAULT_MINIMASS);
            p.out.minimass_skip_loaded = t.get(2) == Some(&"l");
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
                    | Section::Cameras
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
        let (Some(center), Some(back), Some(left)) = (p.node(&t[0]), p.node(&t[1]), p.node(&t[2])) else { continue };
        if p.out.cameras.is_none() {
            p.out.cameras = Some(RefNodes { center, back, left });
        }
    }
    let mut skipped: Vec<_> = p.skipped.into_iter().collect();
    skipped.sort();
    for (s, _) in skipped {
        p.out
            .warnings
            .push(format!("section `{s}` skipped (not simulated)"));
    }
    for &c in &p.out.contacters.clone() {
        p.out.nodes[c as usize].contacter = true;
    }
    p.out
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
    // Wheel spokes from each axle node to the tyre rings (two per ray per axle node:
    // to its own rim side and across), approximated from radius and width.
    for w in &file.wheels {
        let near = w.radius;
        let across = (w.radius * w.radius + w.width * w.width).sqrt();
        for axle in [w.node1, w.node2] {
            halves.push((axle, 0.5 * w.rays as f32 * (near + across)));
        }
    }
    let total: f32 = halves.iter().map(|h| h.1).sum();
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
    let mut file = parse_truck(src);
    distribute_masses(&mut file);
    lower(file)
}

fn lower(file: TruckFile) -> SoftVehicleDesc {
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
        let mut ends: Vec<u32> = [Some(w.node1), Some(w.node2), w.arm].into_iter().flatten().collect();
        let direct = ends.clone();
        for b in desc.beams.iter().filter(|b| b.kind != BeamKind::Hydro) {
            if direct.contains(&b.a) && !ends.contains(&b.b) {
                ends.push(b.b);
            } else if direct.contains(&b.b) && !ends.contains(&b.a) {
                ends.push(b.a);
            }
        }
        for b in desc.beams.iter().filter(|b| b.kind == BeamKind::Hydro && b.hydro_factor != 0.0) {
            let Some(&end) = ends.iter().find(|&&e| b.a == e || b.b == e) else { continue };
            let lever = (p(end) - hub).length().max(0.2);
            let travel = (p(b.a) - p(b.b)).length() * b.hydro_factor.abs();
            steerable = true;
            max_steer = max_steer.max((travel / lever).atan().clamp(10f32.to_radians(), 45f32.to_radians()));
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
        assert!(d.warnings.iter().all(|w| !w.contains("unknown node")), "{:?}", d.warnings);
        assert!(d.refs.is_some());
        assert!(d.wheels[0].steerable, "hydro on the axle node steers the wheel");
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
