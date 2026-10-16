//! Reader for the engine-neutral scene files written by `scenes/export.ts` (format v1: boxes,
//! spheres, ball joints) and `tools/fracture-export` (v2 adds convex-hull compounds and a ground
//! plane; see README "File formats").

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const BODY_WORDS: usize = 24;
pub const JOINT_WORDS: usize = 12;
/// Format v3 joint records (D6 frames, motions, limits, one drive).
pub const JOINT_WORDS_V3: usize = 32;
pub const HULL_WORDS: usize = 2;
const MAGIC: u32 = 0x4353_4252; // "RBSC"

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Half extents in the body frame.
    Box([f32; 3]),
    Sphere(f32),
    /// Convex hulls `first..first+count` of `Scene::hulls` (vertices in the body frame: the
    /// fragment's centre of mass and principal axes). `radius` bounds every vertex.
    Hulls { first: u32, count: u32, radius: f32 },
    /// Static half-space: the plane passes through the body position, its normal is the
    /// body's local +y and its solid side lies below.
    Plane,
}

#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub shape: Shape,
    /// Fixed in place (the ground, pinned net corners). Mass and inertia are zero.
    pub is_static: bool,
    pub mass: f32,
    /// Principal moments in the body frame (uniform-density solid, as three-avbd computes).
    pub inertia: [f32; 3],
    pub friction: f32,
    pub restitution: f32,
    pub position: [f32; 3],
    /// (x, y, z, w)
    pub rotation: [f32; 4],
    pub linvel: [f32; 3],
    pub angvel: [f32; 3],
}

impl Body {
    /// Bounding-sphere radius about the body origin.
    pub fn radius(&self) -> f32 {
        match self.shape {
            Shape::Box(h) => (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt(),
            Shape::Sphere(r) => r,
            Shape::Hulls { radius, .. } => radius,
            Shape::Plane => f32::INFINITY,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointKind {
    /// Hard point-to-point constraint, rotation free.
    Ball,
    /// PhysX-style D6 (format v3): see `D6`.
    D6,
}

/// Per-DOF motion of a D6 joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    Locked,
    Limited,
    Free,
}

/// A D6 joint (format v3). Frames are body-local (anchor + orientation); frame X is the
/// twist / hinge / slider axis. DOF order: X, Y, Z (frame A), twist, swing1 (about Y),
/// swing2 (about Z).
#[derive(Clone, Copy, Debug)]
pub struct D6 {
    pub frame_a: [f32; 4],
    pub frame_b: [f32; 4],
    pub motion: [Motion; 6],
    pub twist: [f32; 2],
    /// Swing limit half-angles about frame Y and Z.
    pub swing: [f32; 2],
    /// Limited swings form one elliptical cone (else two independent angle limits).
    pub cone: bool,
    /// Frame-A X translation limits.
    pub linear_x: [f32; 2],
    pub drive: Option<Drive>,
}

/// One drive (format v3): `stiffness == 0` is a velocity motor, otherwise a position servo.
#[derive(Clone, Copy, Debug)]
pub struct Drive {
    /// 0..2 linear frame-A axis, 3 twist.
    pub dof: u32,
    pub velocity: f32,
    pub force_limit: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub target: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Joint {
    pub kind: JointKind,
    pub body_a: u32,
    pub body_b: u32,
    /// Whether the two joined bodies still collide with each other (false in every suite scene).
    pub collide: bool,
    /// Anchors in each body's frame.
    pub anchor_a: [f32; 3],
    pub anchor_b: [f32; 3],
    /// Present for `JointKind::D6`.
    pub d6: Option<D6>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Protocol {
    pub lead: u32,
    pub rounds: u32,
    pub warm_steps: u32,
    pub batches: u32,
    pub per_batch: u32,
    pub steps: u32,
    pub checkpoints: Vec<u32>,
}

impl Protocol {
    /// The `--smoke` protocol: lead 2, one round of 1 warm step + one batch of 2, checkpoints
    /// at the start and the end. Exercises every code path with almost no compute.
    pub fn smoke(&mut self) {
        (self.lead, self.rounds, self.warm_steps, self.batches, self.per_batch) = (2, 1, 1, 1, 2);
        self.steps = self.lead + self.timed_span();
        self.checkpoints = vec![0, self.steps];
    }

    /// Steps in the timed window, untimed warm steps included.
    pub fn timed_span(&self) -> u32 {
        self.rounds * (self.warm_steps + self.batches * self.per_batch)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Counts {
    pub bodies: usize,
    pub dynamic: usize,
    pub joints: usize,
    pub spheres: usize,
    /// Format v2: convex hulls, and bodies made of more than one hull.
    #[serde(default)]
    pub hulls: usize,
    #[serde(default, rename = "compoundBodies")]
    pub compound_bodies: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Gate {
    pub at: u32,
    pub metric: String,
    pub group: Option<String>,
    pub max: Option<f64>,
    pub min: Option<f64>,
    pub why: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvbdParams {
    pub alpha: f32,
    pub gamma: f32,
    pub beta_lin: f32,
    pub beta_ang: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneMeta {
    pub format: String,
    pub version: u32,
    pub id: String,
    pub tier: String,
    pub dt: f32,
    pub gravity: [f32; 3],
    /// Reference AVBD iteration count (three-avbd's bench case).
    pub iterations: u32,
    pub avbd: AvbdParams,
    pub protocol: Protocol,
    pub counts: Counts,
    pub bin: String,
    pub bin_sha256: String,
    #[serde(default)]
    pub groups: std::collections::BTreeMap<String, Vec<u32>>,
    #[serde(default)]
    pub gates: Vec<Gate>,
    /// Where the scene came from (builder, or asset file + level); adapters that need the
    /// original asset (rubble's baked hull topology) read it from here.
    #[serde(default)]
    pub source: serde_json::Value,
}

pub struct Scene {
    pub path: PathBuf,
    pub meta: SceneMeta,
    pub bodies: Vec<Body>,
    pub joints: Vec<Joint>,
    /// Convex hull vertex lists (format v2), in the owning body's frame.
    pub hulls: Vec<Vec<[f32; 3]>>,
}

impl Scene {
    /// Load `<id>.scene.json` and its `.bin`, verifying the checksum and counts.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let meta: SceneMeta = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(meta.format == "rubble-bench-scene" && (1..=3).contains(&meta.version), "{}: unsupported scene format {} v{}", path.display(), meta.format, meta.version);
        let bin_path = path.with_file_name(&meta.bin);
        let bytes = std::fs::read(&bin_path).with_context(|| format!("reading {}", bin_path.display()))?;
        let digest = hex(&Sha256::digest(&bytes));
        ensure!(digest == meta.bin_sha256, "{}: sha256 {digest} does not match the scene's {}; re-export", bin_path.display(), meta.bin_sha256);
        ensure!(bytes.len() % 4 == 0 && bytes.len() >= 16, "{}: truncated", bin_path.display());
        let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        ensure!(words[0] == MAGIC && (1..=3).contains(&words[1]), "{}: bad header", bin_path.display());
        let version = words[1];
        let (nb, nj) = (words[2] as usize, words[3] as usize);
        // v2 header adds hull and vertex counts; the hull table and vertices follow the joints
        let (head, nh, nv) = if version >= 2 { (6, words[4] as usize, words[5] as usize) } else { (4, 0, 0) };
        let joint_words = if version >= 3 { JOINT_WORDS_V3 } else { JOINT_WORDS };
        ensure!(
            words.len() == head + nb * BODY_WORDS + nj * joint_words + nh * HULL_WORDS + nv * 3,
            "{}: size does not match {nb} bodies, {nj} joints, {nh} hulls, {nv} vertices",
            bin_path.display()
        );
        ensure!(nb == meta.counts.bodies && nj == meta.counts.joints, "{}: counts differ from the scene json", bin_path.display());
        let f = |i: usize| f32::from_bits(words[i]);
        let v3 = |i: usize| [f(i), f(i + 1), f(i + 2)];
        let mut bodies = Vec::with_capacity(nb);
        for b in 0..nb {
            let o = head + b * BODY_WORDS;
            let shape = match words[o] {
                0 => Shape::Box(v3(o + 2)),
                1 => Shape::Sphere(f(o + 2)),
                2 => {
                    let (first, count) = (words[o + 2], words[o + 3]);
                    ensure!(count > 0 && (first + count) as usize <= nh, "body {b}: hulls {first}+{count} out of range");
                    Shape::Hulls { first, count, radius: f(o + 4) }
                }
                3 => {
                    ensure!(words[o + 1] & 1 != 0, "body {b}: a plane must be static");
                    Shape::Plane
                }
                s => bail!("body {b}: unknown shape {s}"),
            };
            bodies.push(Body {
                shape,
                is_static: words[o + 1] & 1 != 0,
                mass: f(o + 5),
                inertia: v3(o + 6),
                friction: f(o + 9),
                position: v3(o + 10),
                rotation: [f(o + 13), f(o + 14), f(o + 15), f(o + 16)],
                linvel: v3(o + 17),
                angvel: v3(o + 20),
                restitution: f(o + 23),
            });
        }
        let base = head + nb * BODY_WORDS;
        let mut joints = Vec::with_capacity(nj);
        for j in 0..nj {
            let o = base + j * joint_words;
            let (a, b) = (words[o], words[o + 1]);
            ensure!((a as usize) < nb && (b as usize) < nb, "joint {j}: body out of range");
            if version < 3 {
                ensure!(words[o + 2] == 0, "joint {j}: unknown kind {}", words[o + 2]);
                joints.push(Joint {
                    kind: JointKind::Ball,
                    body_a: a,
                    body_b: b,
                    collide: words[o + 3] & 1 != 0,
                    anchor_a: v3(o + 4),
                    anchor_b: v3(o + 7),
                    d6: None,
                });
                continue;
            }
            let kind = match words[o + 2] {
                0 => JointKind::Ball,
                1 => JointKind::D6,
                k => bail!("joint {j}: unknown kind {k}"),
            };
            let q = |i: usize| [f(i), f(i + 1), f(i + 2), f(i + 3)];
            let m = words[o + 18];
            let mut motion = [Motion::Locked; 6];
            for (d, slot) in motion.iter_mut().enumerate() {
                *slot = match (m >> (2 * d)) & 3 {
                    0 => Motion::Locked,
                    1 => Motion::Limited,
                    2 => Motion::Free,
                    x => bail!("joint {j}: bad motion {x}"),
                };
            }
            let drive = (words[o + 25] != u32::MAX).then(|| Drive {
                dof: words[o + 25],
                velocity: f(o + 26),
                force_limit: f(o + 27),
                stiffness: f(o + 28),
                damping: f(o + 29),
                target: f(o + 30),
            });
            joints.push(Joint {
                kind,
                body_a: a,
                body_b: b,
                collide: words[o + 3] & 1 != 0,
                anchor_a: v3(o + 4),
                anchor_b: v3(o + 11),
                d6: Some(D6 {
                    frame_a: q(o + 7),
                    frame_b: q(o + 14),
                    motion,
                    twist: [f(o + 19), f(o + 20)],
                    swing: [f(o + 21), f(o + 22)],
                    cone: words[o + 31] & 1 != 0,
                    linear_x: [f(o + 23), f(o + 24)],
                    drive,
                }),
            });
        }
        let hull_base = base + nj * joint_words;
        let vert_base = hull_base + nh * HULL_WORDS;
        let mut hulls = Vec::with_capacity(nh);
        for h in 0..nh {
            let (first, count) = (words[hull_base + h * 2] as usize, words[hull_base + h * 2 + 1] as usize);
            ensure!(count >= 4 && first + count <= nv, "hull {h}: vertices {first}+{count} out of range");
            hulls.push((first..first + count).map(|v| v3(vert_base + v * 3)).collect());
        }
        Ok(Self { path, meta, bodies, joints, hulls })
    }

    pub fn dynamic_count(&self) -> usize {
        self.bodies.iter().filter(|b| !b.is_static).count()
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
