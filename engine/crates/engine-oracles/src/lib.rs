//! Thin oracle adapters — translate `scene_v1` fixtures; do **not** reimplement physics.

use engine_schema::{AvbdParams, JointType, ShapeType};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const THREE_AVBD_REV: &str = "2bb1a318e9aad03640b4bce5b5d222484f0925f7";
pub const AVBD_DEMO3D_REV: &str = "7701bd427d55ca5d03ea1fdf331912ded9169f4b";
pub const NEWTON_REV: &str = "fb92f6d84e4a72796d95cefbb378ba6b6be72f04";
pub const JOLT_REV: &str = "0373ec0dd762e4bc2f6acdb08371ee84fa23c6db";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneV1 {
    pub schema: String,
    pub name: String,
    pub seed: u64,
    pub params: AvbdParams,
    pub bodies: Vec<BodySpec>,
    pub colliders: Vec<ColliderSpec>,
    pub joints: Vec<JointSpec>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodySpec {
    pub id: u32,
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub mass: f32,
    pub inertia: [f32; 3],
    #[serde(default)]
    pub linear_vel: [f32; 3],
    #[serde(default)]
    pub angular_vel: [f32; 3],
    #[serde(default)]
    pub kinematic: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColliderSpec {
    pub id: u32,
    pub body: u32,
    pub shape: ShapeType,
    pub local_pos: [f32; 3],
    pub local_rot: [f32; 4],
    pub params: [f32; 4],
    pub density: f32,
    pub material: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointSpec {
    pub id: u32,
    pub body_a: u32,
    pub body_b: u32,
    pub joint_type: JointType,
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
    pub axis: [f32; 3],
    pub stiffness: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OracleMapping {
    pub oracle: String,
    pub revision: String,
    pub unsupported: Vec<String>,
    pub frame: String,
    pub notes: Vec<String>,
}

pub fn inspiration_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../inspiration/pinned")
}

pub fn three_avbd_mapping() -> OracleMapping {
    OracleMapping {
        oracle: "three-avbd".into(),
        revision: THREE_AVBD_REV.into(),
        unsupported: vec![
            "soft-body".into(),
            "cloth".into(),
            "plastic-yield".into(),
            "wind".into(),
        ],
        frame: "oracle=Z-up; engine=Y-up (x,y,z)->(x,z,y)".into(),
        notes: vec![
            "Primary 3D GPU AVBD numerical / seeded oracle".into(),
            "Match iterations, alpha=0.99, gamma=0.999, beta_lin/ang".into(),
        ],
    }
}

pub fn avbd_demo3d_mapping() -> OracleMapping {
    OracleMapping {
        oracle: "avbd-demo3d".into(),
        revision: AVBD_DEMO3D_REV.into(),
        unsupported: vec!["gpu-scale".into(), "convex-hull".into()],
        frame: "oracle=Z-up; engine=Y-up".into(),
        notes: vec![
            "Official demo parameter & residual tie-break".into(),
            format!("gravity demo Z={} → engine Y", -10.0),
        ],
    }
}

pub fn newton_mapping() -> OracleMapping {
    OracleMapping {
        oracle: "newton".into(),
        revision: NEWTON_REV.into(),
        unsupported: vec![
            "distance-joint-in-vbd".into(),
            "record compliant-ALM vs legacy when comparing".into(),
        ],
        frame: "engine=Y-up".into(),
        notes: vec![
            "Joint semantics oracle (BALL/FIXED/REVOLUTE/PRISMATIC/D6)".into(),
            "Do not compare raw stiffness across formulations".into(),
        ],
    }
}

pub fn jolt_mapping() -> OracleMapping {
    OracleMapping {
        oracle: "JoltPhysics".into(),
        revision: JOLT_REV.into(),
        unsupported: vec!["avbd-multipliers".into()],
        frame: "engine=Y-up".into(),
        notes: vec![
            "Independent rigid-body / sleep / LinearCast CCD behavior".into(),
            "Physical envelopes only, not bit-identical poses".into(),
        ],
    }
}

/// Convert a demo Z-up position into engine Y-up.
pub fn z_up_to_y_up(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, p.y)
}

pub fn load_scene_json(path: &Path) -> anyhow::Result<SceneV1> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn write_mapping(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, m) in [
        ("three_avbd.json", three_avbd_mapping()),
        ("avbd_demo3d.json", avbd_demo3d_mapping()),
        ("newton_vbd.json", newton_mapping()),
        ("jolt.json", jolt_mapping()),
    ] {
        let path = dir.join(name);
        std::fs::write(path, serde_json::to_string_pretty(&m)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Developer-machine check only: inspiration/ is gitignored and absent in
    /// CI and worktrees. Run with `cargo test -p engine-oracles -- --ignored`.
    #[test]
    #[ignore = "needs the gitignored inspiration/pinned checkout"]
    fn pinned_repos_present() {
        let root = inspiration_root();
        for name in [
            "three-avbd",
            "avbd-demo3d",
            "avbd-sandbox",
            "beat-box",
            "newton",
            "JoltPhysics",
        ] {
            assert!(root.join(name).is_dir(), "missing {name} under {:?}", root);
        }
    }

    #[test]
    fn mappings_serialize() {
        let m = three_avbd_mapping();
        assert_eq!(m.revision, THREE_AVBD_REV);
        let _ = serde_json::to_string(&m).unwrap();
    }
}
