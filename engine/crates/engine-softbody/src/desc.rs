//! `SoftVehicleDesc`: the engine-neutral intermediate form both importers lower
//! into (nodes, beams with resolved properties, collision triangles, wheels,
//! drivetrain). Positions are in Rubble's frame (Y up, metres).

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// JSON has no infinity: write ±∞ thresholds as ±f32::MAX (which every
/// comparison treats the same) and read `null` back as +∞.
mod inf_f32 {
    use super::*;
    pub fn serialize<S: Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f32(if v.is_infinite() { f32::MAX.copysign(*v) } else { *v })
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
        Ok(Option::<f32>::deserialize(d)?.unwrap_or(f32::INFINITY))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeamKind {
    /// ½k(ℓ − L)² with axial damping; yields and breaks.
    #[default]
    Normal,
    /// Compression only (RoR `s`, JBeam SUPPORT); breaks past a long bound.
    Support,
    /// Tension only (RoR `r`, ropes).
    Rope,
    /// Spring-damper inside [short, long] bounds, stiff limit spring outside
    /// (RoR shocks, JBeam BOUNDED). Never deforms.
    Bounded,
    /// Rest length follows an input: L = L₀(1 − input·factor). Never deforms.
    Hydro,
}

impl BeamKind {
    pub fn code(self) -> u32 {
        match self {
            BeamKind::Normal => 0,
            BeamKind::Support => 1,
            BeamKind::Rope => 2,
            BeamKind::Bounded => 3,
            BeamKind::Hydro => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftNode {
    pub name: String,
    pub position: [f32; 3],
    /// kg (after the format's mass rules).
    pub mass: f32,
    /// Gets a sphere collider (world and other vehicles).
    pub collision: bool,
    /// Immovable (RoR `fixes`, JBeam `fixed`).
    pub fixed: bool,
    /// Contacter: collides with other vehicles' triangles.
    pub contacter: bool,
    pub friction: f32,
    /// Part of the rigid core attachment set (filled by the lowering).
    pub frame: bool,
}

impl Default for SoftNode {
    fn default() -> Self {
        Self {
            name: String::new(),
            position: [0.0; 3],
            mass: 0.0,
            collision: true,
            fixed: false,
            contacter: false,
            friction: 0.8,
            frame: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftBeam {
    pub a: u32,
    pub b: u32,
    pub kind: BeamKind,
    /// N/m.
    pub spring: f32,
    /// N·s/m.
    pub damp: f32,
    /// Deform (yield) threshold, N. `f32::INFINITY` = elastic only.
    #[serde(with = "inf_f32")]
    pub deform: f32,
    /// Break strength, N. `f32::INFINITY` = unbreakable.
    #[serde(with = "inf_f32")]
    pub strength: f32,
    /// RoR plastic coefficient 0..1 (0 = the beam ends exactly on its threshold).
    pub plastic_coef: f32,
    /// Bounded / support bounds as fractions of L₀ (short = compression, long = extension).
    pub short_bound: f32,
    pub long_bound: f32,
    /// Rest length scale at spawn (RoR shock precompression, JBeam beamPrecompression).
    pub precompression: f32,
    /// Hydro: L = L₀(1 − input·factor).
    pub hydro_factor: f32,
    /// Break group (0 = none). Breaking any beam of a group breaks the whole group.
    pub break_group: u32,
    pub visible: bool,
}

impl Default for SoftBeam {
    fn default() -> Self {
        Self {
            a: 0,
            b: 0,
            kind: BeamKind::Normal,
            spring: 9.0e6,
            damp: 12000.0,
            deform: 400_000.0,
            strength: 1.0e6,
            plastic_coef: 0.0,
            short_bound: 0.0,
            long_bound: 0.0,
            precompression: 1.0,
            hydro_factor: 0.0,
            break_group: 0,
            visible: true,
        }
    }
}

/// Collision (and render) triangle, counter-clockwise front face.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftTriangle {
    pub nodes: [u32; 3],
    pub collision: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftWheel {
    /// Axle node pair (n1 → n2 is the axle).
    pub node1: u32,
    pub node2: u32,
    /// Rigidity / steering arm node (`None` = none).
    pub arm: Option<u32>,
    pub radius: f32,
    pub width: f32,
    pub mass: f32,
    pub braked: bool,
    /// 0 = free, 1 = forward, -1 = backward.
    pub propulsion: i32,
    pub steerable: bool,
    pub max_steer: f32,
}

impl Default for SoftWheel {
    fn default() -> Self {
        Self {
            node1: 0,
            node2: 0,
            arm: None,
            radius: 0.35,
            width: 0.2,
            mass: 20.0,
            braked: true,
            propulsion: 0,
            steerable: false,
            max_steer: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftEngine {
    pub shift_down_rpm: f32,
    pub shift_up_rpm: f32,
    pub max_torque: f32,
    /// Differential (global) ratio.
    pub diff_ratio: f32,
    pub reverse_ratio: f32,
    pub gear_ratios: Vec<f32>,
    pub max_rpm: f32,
}

impl Default for SoftEngine {
    fn default() -> Self {
        Self {
            shift_down_rpm: 1500.0,
            shift_up_rpm: 4500.0,
            max_torque: 400.0,
            diff_ratio: 3.5,
            reverse_ratio: 3.2,
            gear_ratios: vec![3.4, 2.2, 1.5, 1.1, 0.9],
            max_rpm: 6000.0,
        }
    }
}

/// Vehicle reference frame from the file (RoR `cameras`, JBeam `refNodes`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RefNodes {
    pub center: u32,
    pub back: u32,
    pub left: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SoftVehicleDesc {
    pub name: String,
    pub nodes: Vec<SoftNode>,
    pub beams: Vec<SoftBeam>,
    pub triangles: Vec<SoftTriangle>,
    pub wheels: Vec<SoftWheel>,
    pub engine: Option<SoftEngine>,
    pub brake_torque: Option<f32>,
    pub refs: Option<RefNodes>,
    /// Triangle collision thickness δ (RoR `set_collision_range`).
    pub collision_range: f32,
    /// Sections or options the importer skipped.
    pub warnings: Vec<String>,
}

impl SoftVehicleDesc {
    pub fn total_mass(&self) -> f32 {
        self.nodes.iter().map(|n| n.mass).sum()
    }

    pub fn rest_length(&self, beam: &SoftBeam) -> f32 {
        let a = glam::Vec3::from_array(self.nodes[beam.a as usize].position);
        let b = glam::Vec3::from_array(self.nodes[beam.b as usize].position);
        (a - b).length()
    }

    /// JSON form (infinite thresholds as ±f32::MAX), e.g. for hosts that pass
    /// descriptors across a language boundary.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("desc serialises")
    }

    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut mn = glam::Vec3::splat(f32::MAX);
        let mut mx = glam::Vec3::splat(f32::MIN);
        for n in &self.nodes {
            let p = glam::Vec3::from_array(n.position);
            mn = mn.min(p);
            mx = mx.max(p);
        }
        (mn.to_array(), mx.to_array())
    }

    /// Validate and repair (design step 3): drop zero-length and duplicate beams
    /// and beams with out-of-range nodes, flag isolated nodes, apply a mass floor.
    pub fn repair(&mut self, min_mass: f32) {
        let n = self.nodes.len() as u32;
        let mut seen = std::collections::HashSet::new();
        let before = self.beams.len();
        let nodes = &self.nodes;
        self.beams.retain(|b| {
            if b.a >= n || b.b >= n || b.a == b.b {
                return false;
            }
            let pa = glam::Vec3::from_array(nodes[b.a as usize].position);
            let pb = glam::Vec3::from_array(nodes[b.b as usize].position);
            if (pa - pb).length() < 1.0e-4 {
                return false;
            }
            seen.insert((b.a.min(b.b), b.a.max(b.b), b.kind.code()))
        });
        let dropped = before - self.beams.len();
        if dropped > 0 {
            self.warnings.push(format!("dropped {dropped} duplicate, zero-length or invalid beams"));
        }
        self.triangles.retain(|t| t.nodes.iter().all(|&i| i < n) && t.nodes[0] != t.nodes[1] && t.nodes[1] != t.nodes[2] && t.nodes[0] != t.nodes[2]);
        let mut degree = vec![0u32; self.nodes.len()];
        for b in &self.beams {
            degree[b.a as usize] += 1;
            degree[b.b as usize] += 1;
        }
        let isolated = degree.iter().filter(|&&d| d == 0).count();
        if isolated > 0 {
            self.warnings.push(format!("{isolated} isolated nodes (no beams)"));
        }
        for node in &mut self.nodes {
            if node.mass < min_mass {
                node.mass = min_mass;
            }
        }
    }
}
