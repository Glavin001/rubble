//! Wrecking Yard: tall structures for heavy balls, built here and written as scene packs
//! (the JSON `scene_pack` imports), so the Fracture lab loads them as it loads any pack:
//! `engine-fracture` `pack::build_pack_scene` for its own modes, stress-web
//! `stress_set_pack` for the stress modes.
//!
//! * [`frame10`]: a 10-storey reinforced-concrete frame, 3 x 3 bays of 6 m: columns,
//!   beam-column joints, beams (two chunks a span) and slabs (four chunks a bay), on pad
//!   footings. Seeded from `builders::frame_column_removal` (`grid`/`chunk` boxes and
//!   `auto_bonds`).
//! * [`round_tower`]: eight round floors on a central core and a ring of eight round
//!   columns, the red-sphere target.
//! * [`brick_tower`]: a square brick shaft in running bond with interlocked corners.
//!
//! Materials are the calibrated ones (`material.rs`): the pack's materials carry their
//! strengths and stiffness, and each node its mass at the calibrated density, so the
//! profiles' authored import (`Profile::import_options`: the pack's fatal limits, masses
//! and modulus) gives the calibrated values. Reinforced concrete gets the code-minimum
//! reinforcement of every pack import. Footings are the supports (mass 0). The weight is
//! carried as in every pack: the stress solve takes gravity from the first frame.
//!
//! Packs are Y-up and the builders Z-up (`scene_pack`'s map `(x, y, z) -> (x, -z, y)`,
//! inverted here). Every chunk sits at or above grade (z = 0).
//!
//! [`BALLS`] are the yard's ball presets and [`launch`] the ready-made shot along a
//! pack's line of attack (`defaults.wreckingYard`, read with [`aim_of`]), as the lab's
//! row shot is for its rows.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::builders::{auto_bonds, chunk};
use crate::hull::{contact_patch, ConvexHull};
use crate::material::Material;
use crate::math::Vec3;
use crate::scene::{BondDesc, ChunkDesc, ImpactorDesc, ImpactorShape, Support};

/// Faces closer than this (m) are touching (the builders' faces meet exactly).
const SEAM: f64 = 1e-6;

/// The step Blast's impulses are accumulated over (vibe-city's 60 Hz; `engine-fracture`
/// `pack::BLAST_DT`).
const BLAST_DT: f64 = 1.0 / 60.0;

/// The yard's structures: (key, title), in menu order.
pub const STRUCTURES: &[(&str, &str)] = &[("wy_frame10", "10-floor concrete frame"), ("wy_round_tower", "Round tower on columns"), ("wy_brick_tower", "Brick tower")];

/// A structure as the builders make it (Z-up): chunks (box or hull, `material` the pack's
/// asset material, footings `Support::Fixed`), bonds (`material` the pack material's
/// name), the line a ball comes in along, and the click ball.
pub struct Structure {
    pub key: &'static str,
    pub title: &'static str,
    pub chunks: Vec<ChunkDesc>,
    pub bonds: Vec<BondDesc>,
    pub aim: Aim,
    pub ball: Ball,
}

/// Where a ready-made shot goes (Z-up, the stress scene's frame): the point it is aimed
/// at and the horizontal direction it travels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aim {
    pub target: [f64; 3],
    pub direction: [f64; 3],
}

/// A ball preset.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Ball {
    pub key: &'static str,
    /// Menu entry.
    pub menu: &'static str,
    /// kg.
    pub mass: f64,
    /// m.
    pub radius: f64,
    /// m/s.
    pub speed: f64,
    /// Rolls in along the ground (centre at its radius, spinning without slip) instead of
    /// flying at the target's height.
    pub rolling: bool,
}

/// Radii of solid steel balls of 10 t and 50 t: (3 m / (4 pi rho))^(1/3) at the
/// calibrated steel density (7850 kg/m^3; `wrecking_balls_are_solid_steel` checks them).
const BALL_10T_RADIUS: f64 = 0.6725;
const BALL_50T_RADIUS: f64 = 1.1499;

/// The demolition sphere: a 15 m steel shell, 5 cm thick (4 pi r^2 t rho = 277 t).
const SPHERE_RADIUS: f64 = 7.5;
const SPHERE_SHELL: f64 = 0.05;

/// The yard's ball presets, in menu order. Solid steel wrecking balls fly in at the
/// target's height; the demolition sphere rolls in along the ground. The lab's row shot
/// (200 t at 40 m/s) is the model.
pub const BALLS: &[Ball] = &[
    Ball { key: "ball10t", menu: "10 t wrecking ball · 40 m/s", mass: 10_000.0, radius: BALL_10T_RADIUS, speed: 40.0, rolling: false },
    Ball { key: "ball50t", menu: "50 t wrecking ball · 20 m/s", mass: 50_000.0, radius: BALL_50T_RADIUS, speed: 20.0, rolling: false },
    Ball {
        key: "sphere15m",
        menu: "15 m demolition sphere · 277 t, rolls in at 12 m/s",
        mass: 4.0 * std::f64::consts::PI * SPHERE_RADIUS * SPHERE_RADIUS * SPHERE_SHELL * 7850.0,
        radius: SPHERE_RADIUS,
        speed: 12.0,
        rolling: true,
    },
];

/// A ball preset by key.
pub fn ball(key: &str) -> Option<&'static Ball> {
    BALLS.iter().find(|b| b.key == key)
}

/// Clear space between a ball and its target when it is launched (m).
const STANDOFF: f64 = 6.0;

/// The shot of `ball` along `aim` (Z-up, ground at z = 0): launched `STANDOFF` m short of
/// touching the target, at the target's height (with the lift that brings it there under
/// gravity), or rolling along the ground.
pub fn launch(ball: &Ball, aim: &Aim, name: &str) -> ImpactorDesc {
    let d = Vec3::from_array(aim.direction).normalized();
    let t = Vec3::from_array(aim.target);
    let distance = ball.radius + STANDOFF;
    let mut p = t - d * distance;
    let mut v = d * ball.speed;
    let mut w = Vec3::ZERO;
    if ball.rolling {
        p.z = ball.radius;
        // No slip at the ground: v + w x (-r up) = 0.
        w = Vec3::new(0.0, 0.0, 1.0).cross(v) * (1.0 / ball.radius);
    } else {
        p.z = t.z.max(ball.radius);
        v.z += 0.5 * 9.81 * distance / ball.speed;
    }
    ImpactorDesc {
        name: name.into(),
        shape: ImpactorShape::Sphere { radius: ball.radius },
        mass: ball.mass,
        position: p.to_array(),
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: v.to_array(),
        angular_velocity: w.to_array(),
        material: "projectile".into(),
        crush: None,
    }
}

/// A structure by key.
pub fn structure(key: &str) -> Option<Structure> {
    match key {
        "wy_frame10" => Some(frame10()),
        "wy_round_tower" => Some(round_tower()),
        "wy_brick_tower" => Some(brick_tower()),
        _ => None,
    }
}

/// A structure's scene pack, by key.
pub fn pack(key: &str) -> Option<Value> {
    structure(key).map(|s| to_pack(&s))
}

/// The line of attack a pack carries (`defaults.wreckingYard`), in the stress scene's
/// frame (Z-up); None for a pack without one.
pub fn aim_of(pack: &Value) -> Option<Aim> {
    let a = &pack["defaults"]["wreckingYard"];
    let v = |v: &Value| -> Option<[f64; 3]> { Some(z_up([v["x"].as_f64()?, v["y"].as_f64()?, v["z"].as_f64()?])) };
    Some(Aim { target: v(&a["target"])?, direction: v(&a["direction"])? })
}

/// Z-up to the packs' Y-up (the inverse of `scene_pack`'s `(x, y, z) -> (x, -z, y)`).
fn y_up(v: [f64; 3]) -> [f64; 3] {
    [v[0], v[2], -v[1]]
}

fn z_up(v: [f64; 3]) -> [f64; 3] {
    [v[0], -v[2], v[1]]
}

fn xyz(v: [f64; 3]) -> Value {
    let p = y_up(v);
    json!({"x": p[0], "y": p[1], "z": p[2]})
}

/// The calibrated material an asset material stands for (`scene_pack`'s mapping): its
/// chunks' density and its joints' strengths and stiffness.
fn calibrated(asset: &str) -> (Material, Material) {
    match asset {
        "reinforced-concrete" | "footing-anchor" => (Material::concrete(), Material::concrete()),
        "brick" => (Material::brick(), Material::mortar()),
        other => panic!("no calibrated material for {other}"),
    }
}

/// A chunk's volume: its hull's, else its box's.
pub fn volume(c: &ChunkDesc) -> f64 {
    match &c.hull {
        Some(points) => ConvexHull::new(&points.iter().map(|p| Vec3::from_array(*p)).collect::<Vec<_>>()).expect("a valid hull").unit_mass().volume,
        None => 8.0 * c.half_extents.iter().product::<f64>(),
    }
}

/// A chunk's mass at its calibrated density (0 for a support).
pub fn mass(c: &ChunkDesc) -> f64 {
    if c.support == Support::Fixed {
        0.0
    } else {
        calibrated(&c.material).0.density * volume(c)
    }
}

/// The scene pack of a structure: nodes (centroid, mass, colliders, materials, types),
/// bonds (the builders' patches) and the pack's materials (one per joint material, the
/// calibrated strengths as both its elastic and fatal limits).
pub fn to_pack(s: &Structure) -> Value {
    // Pack materials by joint material name, in order of first use.
    let mut index: BTreeMap<String, usize> = BTreeMap::new();
    let mut materials = Vec::new();
    for b in &s.bonds {
        if index.contains_key(&b.material) {
            continue;
        }
        let (bulk, joint) = calibrated(&b.material);
        index.insert(b.material.clone(), materials.len());
        materials.push(json!({
            "name": b.material,
            "tensionElastic": joint.tensile_strength,
            "tensionFatal": joint.tensile_strength,
            "compressionElastic": joint.compressive_strength,
            "compressionFatal": joint.compressive_strength,
            "shearElastic": joint.cohesion,
            "shearFatal": joint.cohesion,
            "elasticModulus": joint.youngs_modulus,
            "density": bulk.density,
        }));
    }
    let nodes: Vec<Value> = s.chunks.iter().map(|c| json!({"centroid": xyz(c.center), "mass": mass(c), "volume": volume(c)})).collect();
    let sizes: Vec<Value> = s.chunks.iter().map(|c| xyz_abs(c.half_extents.map(|h| 2.0 * h))).collect();
    let colliders: Vec<Value> = s
        .chunks
        .iter()
        .map(|c| match &c.hull {
            Some(points) => json!({"kind": "convex_hull", "points": points.iter().flat_map(|p| y_up(*p)).collect::<Vec<f64>>()}),
            None => json!({"kind": "cuboid", "halfExtents": xyz_abs(c.half_extents)}),
        })
        .collect();
    let bonds: Vec<Value> = s.bonds.iter().map(|b| json!({"node0": b.a, "node1": b.b, "centroid": xyz(b.centroid), "normal": xyz(b.normal), "area": b.area, "m": index[&b.material]})).collect();
    let (lo, hi) = bounds(&s.chunks);
    let height = hi[2] - lo[2];
    // The pack-wide limits Blast-style readers take (`engine-fracture` `pack.rs`, the lab's
    // D6 bonds): Blast's bond stress is an impulse over one 60 Hz step per area, so a
    // strength of S Pa is a limit of S / 60. The structure's one joint material's.
    let (_, joint) = calibrated(&s.bonds[0].material);
    assert!(s.bonds.iter().all(|b| b.material == s.bonds[0].material), "one joint material a structure");
    let limits = json!({
        "tensionFatal": joint.tensile_strength * BLAST_DT,
        "compressionFatal": joint.compressive_strength * BLAST_DT,
        "shearFatal": joint.cohesion * BLAST_DT,
    });
    json!({
        "version": 2,
        "key": s.key,
        "title": s.title,
        "defaults": {
            "camera": {"target": {"x": 0.0, "y": 0.5 * height, "z": 0.0}, "distance": 2.2 * height},
            "projectile": {"radius": s.ball.radius, "mass": s.ball.mass, "speed": s.ball.speed, "ttlMs": 8000},
            "solver": {"gravity": -9.81, "materialScale": 1, "materials": materials, "limits": limits},
            "wreckingYard": {"target": xyz(s.aim.target), "direction": xyz(s.aim.direction)},
        },
        "scenario": {
            "nodes": nodes,
            "bonds": bonds,
            "nodeSizes": sizes,
            "nodeColliders": colliders,
            "nodeMaterials": s.chunks.iter().map(|c| c.material.clone()).collect::<Vec<_>>(),
            "nodeTypes": s.chunks.iter().map(|c| c.groups.first().cloned().unwrap_or_default()).collect::<Vec<_>>(),
            "nodeGroups": s.chunks.iter().map(|c| c.groups.get(1).cloned().unwrap_or_default()).collect::<Vec<_>>(),
        },
    })
}

/// Extents (Y-up, unsigned).
fn xyz_abs(v: [f64; 3]) -> Value {
    json!({"x": v[0], "y": v[2], "z": v[1]})
}

/// The chunks' bounding box (Z-up).
pub fn bounds(chunks: &[ChunkDesc]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in chunks {
        for k in 0..3 {
            lo[k] = lo[k].min(c.center[k] - c.half_extents[k]);
            hi[k] = hi[k].max(c.center[k] + c.half_extents[k]);
        }
    }
    (lo, hi)
}

/// A box chunk spanning `lo..hi`, of type `kind` in `group`.
fn block(lo: [f64; 3], hi: [f64; 3], material: &str, kind: &str, group: String) -> ChunkDesc {
    let c = Vec3::new(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]));
    let h = Vec3::new(0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1]), 0.5 * (hi[2] - lo[2]));
    let mut ch = chunk(c, h, material);
    ch.groups = vec![kind.into(), group];
    ch
}

/// A hull chunk of `points` (world, Z-up), centred on their mean.
fn hull_chunk(points: &[Vec3], material: &str, kind: &str, group: String) -> ChunkDesc {
    let c = points.iter().fold(Vec3::ZERO, |a, p| a + *p) * (1.0 / points.len() as f64);
    let local: Vec<Vec3> = points.iter().map(|p| *p - c).collect();
    let half = local.iter().fold(Vec3::ZERO, |a, p| a.component_max(p.abs()));
    let mut ch = chunk(c, half, material);
    ch.hull = Some(local.iter().map(|p| p.to_array()).collect());
    ch.groups = vec![kind.into(), group];
    ch
}

/// The bonds that hold something: none between two supports (both are fixed).
fn between_free(chunks: &[ChunkDesc], mut bonds: Vec<BondDesc>) -> Vec<BondDesc> {
    bonds.retain(|b| chunks[b.a].support != Support::Fixed || chunks[b.b].support != Support::Fixed);
    bonds
}

/// A footing: the support (mass 0) a structure stands on.
fn footing(mut c: ChunkDesc) -> ChunkDesc {
    c.support = Support::Fixed;
    c
}

/// Bonds between every pair of hull (or box) chunks with faces touching over an area:
/// the largest shared patch of each pair (`hull::contact_patch`), named by the chunks'
/// joint material (`joint`).
fn face_bonds(chunks: &[ChunkDesc], joint: impl Fn(&ChunkDesc, &ChunkDesc) -> String) -> Vec<BondDesc> {
    let hulls: Vec<ConvexHull> = chunks
        .iter()
        .map(|c| {
            let points: Vec<Vec3> = match &c.hull {
                Some(p) => p.iter().map(|p| Vec3::from_array(*p)).collect(),
                None => {
                    let h = Vec3::from_array(c.half_extents);
                    (0..8).map(|k| Vec3::new(if k & 1 == 0 { -h.x } else { h.x }, if k & 2 == 0 { -h.y } else { h.y }, if k & 4 == 0 { -h.z } else { h.z })).collect()
                }
            };
            ConvexHull::new(&points).expect("a valid hull")
        })
        .collect();
    let reach: Vec<f64> = chunks.iter().map(|c| Vec3::from_array(c.half_extents).norm()).collect();
    let mut bonds = Vec::new();
    for i in 0..chunks.len() {
        let ca = Vec3::from_array(chunks[i].center);
        for j in i + 1..chunks.len() {
            let cb = Vec3::from_array(chunks[j].center);
            if (cb - ca).norm() > reach[i] + reach[j] + SEAM {
                continue;
            }
            let best = hulls[i]
                .faces
                .iter()
                .filter(|f| f.normal.dot(cb - ca) > 0.0)
                .filter_map(|f| contact_patch(&hulls[i], ca, &hulls[j], cb, f.normal, SEAM))
                .filter(|p| p.width[0] > 1e-6 && p.width[1] > 1e-6)
                .max_by(|x, y| x.area.total_cmp(&y.area));
            if let Some(p) = best {
                bonds.push(BondDesc {
                    a: i,
                    b: j,
                    centroid: p.centroid.to_array(),
                    normal: p.normal.to_array(),
                    tangent: p.tangent.to_array(),
                    area: p.area,
                    width: p.width,
                    material: joint(&chunks[i], &chunks[j]),
                    rebar: None,
                    buckling_length: None,
                    level: 0,
                    groups: Vec::new(),
                    derived: None,
                });
            }
        }
    }
    bonds
}

/// The joint material between two chunks: mortar where a brick meets anything, else
/// the concrete's.
fn joint_of(a: &ChunkDesc, b: &ChunkDesc) -> String {
    if a.material == "brick" || b.material == "brick" {
        "brick".into()
    } else {
        "reinforced-concrete".into()
    }
}

/// The 10-storey frame's dimensions (m).
pub struct FrameLayout;

impl FrameLayout {
    pub const BAYS: usize = 3;
    pub const BAY: f64 = 6.0;
    pub const STOREYS: usize = 10;
    pub const STOREY: f64 = 3.5;
    /// Square column section.
    pub const COLUMN: f64 = 0.5;
    /// Beam depth (with the slab) and width.
    pub const BEAM_DEPTH: f64 = 0.6;
    pub const BEAM_WIDTH: f64 = 0.5;
    pub const SLAB: f64 = 0.25;
    /// The core's walls (the central bay's four sides, under its beams) and their strip
    /// footings.
    pub const WALL: f64 = 0.3;
    /// Pad footing: plan side and depth (its top is the ground storey's base).
    pub const FOOTING: f64 = 1.0;
    pub const FOOTING_DEPTH: f64 = 0.5;
    /// Chunks per storey: 16 joints, 16 columns, 48 beam halves, 36 slab quarters, the
    /// core's 8 wall halves.
    pub const PER_STOREY: usize = 16 + 16 + 48 + 36 + 8;
    /// The storeys, 16 pad footings and the core's 8 strip footings.
    pub const CHUNKS: usize = Self::STOREYS * Self::PER_STOREY + 16 + 8;
}

/// A 10-storey reinforced-concrete frame (3 x 3 bays of 6 m, 3.5 m storeys): per storey
/// a 0.5 m column on every grid line, a 0.5 x 0.6 m beam-column joint above it, beams
/// between the joints in two chunks a span and a 0.25 m slab in four chunks a bay, on
/// 1 m pad footings; the central bay is the lift core, walled on its four sides (0.3 m
/// walls between the columns, under the beams, two chunks a side, on strip footings).
/// 1264 chunks. The ball comes along the front column line (y = -9) at the corner
/// column's ground storey.
pub fn frame10() -> Structure {
    type L = FrameLayout;
    let n = L::BAYS + 1;
    let half_span = 0.5 * L::BAYS as f64 * L::BAY;
    let line = |i: usize| i as f64 * L::BAY - half_span;
    let (hc, hw) = (0.5 * L::COLUMN, 0.5 * L::BEAM_WIDTH);
    let base = L::FOOTING_DEPTH;
    let top = |s: usize| base + (s + 1) as f64 * L::STOREY;
    let rc = "reinforced-concrete";
    let mut chunks = Vec::with_capacity(L::CHUNKS);
    for j in 0..n {
        for i in 0..n {
            let (x, y) = (line(i), line(j));
            let h = 0.5 * L::FOOTING;
            chunks.push(footing(block([x - h, y - h, 0.0], [x + h, y + h, base], "footing-anchor", "footing", "footing".into())));
        }
    }
    // The core's walls run along the grid lines 1 and 2 between their columns: each
    // side's two halves (from, to) along the line, on line `c`, running along x or y.
    let (core0, core1) = (line(1), line(2));
    let mid = 0.5 * (core0 + core1);
    let core_sides = [(core0, true), (core1, true), (core0, false), (core1, false)];
    let wall = |u0: f64, u1: f64, c: f64, along_x: bool, z0: f64, z1: f64, kind: &str, group: String, material: &str| {
        let hw = 0.5 * L::WALL;
        if along_x {
            block([u0, c - hw, z0], [u1, c + hw, z1], material, kind, group)
        } else {
            block([c - hw, u0, z0], [c + hw, u1, z1], material, kind, group)
        }
    };
    let hf = 0.5 * L::FOOTING;
    for &(c, along_x) in &core_sides {
        for (u0, u1) in [(core0 + hf, mid), (mid, core1 - hf)] {
            chunks.push(footing(wall(u0, u1, c, along_x, 0.0, base, "footing", "footing".into(), "footing-anchor")));
        }
    }
    for s in 0..L::STOREYS {
        let (z1, floor) = (top(s), format!("storey-{}", s + 1));
        let z0 = if s == 0 { base } else { top(s - 1) };
        let zb = z1 - L::BEAM_DEPTH;
        for j in 0..n {
            for i in 0..n {
                let (x, y) = (line(i), line(j));
                chunks.push(block([x - hc, y - hc, z0], [x + hc, y + hc, zb], rc, "column", floor.clone()));
                chunks.push(block([x - hc, y - hc, zb], [x + hc, y + hc, z1], rc, "joint", floor.clone()));
            }
        }
        // Beams along x (on each y line) and along y (on each x line), two halves a span.
        for k in 0..n {
            for i in 0..L::BAYS {
                let (a, b) = (line(i) + hc, line(i + 1) - hc);
                let m = 0.5 * (a + b);
                for (u0, u1) in [(a, m), (m, b)] {
                    let c = line(k);
                    chunks.push(block([u0, c - hw, zb], [u1, c + hw, z1], rc, "beam", floor.clone()));
                    chunks.push(block([c - hw, u0, zb], [c + hw, u1, z1], rc, "beam", floor.clone()));
                }
            }
        }
        for &(c, along_x) in &core_sides {
            for (u0, u1) in [(core0 + hc, mid), (mid, core1 - hc)] {
                chunks.push(wall(u0, u1, c, along_x, z0, zb, "wall", floor.clone(), rc));
            }
        }
        // Slabs between the beams, four quarters a bay, flush with the beams' tops.
        for j in 0..L::BAYS {
            for i in 0..L::BAYS {
                let (x0, x1, y0, y1) = (line(i) + hw, line(i + 1) - hw, line(j) + hw, line(j + 1) - hw);
                let (xm, ym) = (0.5 * (x0 + x1), 0.5 * (y0 + y1));
                for (a, b) in [(x0, xm), (xm, x1)] {
                    for (c, d) in [(y0, ym), (ym, y1)] {
                        chunks.push(block([a, c, z1 - L::SLAB], [b, d, z1], rc, "slab", floor.clone()));
                    }
                }
            }
        }
    }
    let bonds = between_free(&chunks, auto_bonds(&chunks, joint_of));
    let corner = line(0);
    Structure {
        key: "wy_frame10",
        title: "10-floor concrete frame",
        chunks,
        bonds,
        aim: Aim { target: [corner - hc, corner, base + 0.5 * (L::STOREY - L::BEAM_DEPTH)], direction: [1.0, 0.0, 0.0] },
        ball: BALLS[0],
    }
}

/// The round tower's dimensions (m).
pub struct RoundLayout;

impl RoundLayout {
    pub const FLOORS: usize = 8;
    pub const STOREY: f64 = 3.5;
    pub const SECTORS: usize = 16;
    /// Core, inner ring and outer (floor edge) radii, to the polygons' corners.
    pub const CORE: f64 = 1.5;
    pub const RING: f64 = 4.5;
    pub const EDGE: f64 = 8.0;
    pub const SLAB: f64 = 0.3;
    /// Round columns: count, radius of their circle, their own radius (octagons).
    pub const COLUMNS: usize = 8;
    pub const COLUMN_AT: f64 = 6.25;
    pub const COLUMN: f64 = 0.35;
    pub const FOOTING_DEPTH: f64 = 0.5;
    /// Chunks per floor: a core segment, two rings of slab sectors, the columns under them.
    pub const PER_FLOOR: usize = 1 + 2 * Self::SECTORS + Self::COLUMNS;
    pub const CHUNKS: usize = Self::FLOORS * Self::PER_FLOOR + 1 + Self::COLUMNS;
}

/// Eight round floors (8 m radius, 0.3 m slabs in 32 sectors) on a 16-sided core and a
/// ring of eight round (octagonal) columns, on footings: the red-sphere target. 337
/// chunks. The ball comes in along +x at the column nearest -x (behind it, the core), at
/// the first storey.
pub fn round_tower() -> Structure {
    type L = RoundLayout;
    let n = L::SECTORS;
    let corner = |k: usize, r: f64| {
        let a = 2.0 * std::f64::consts::PI * (k % n) as f64 / n as f64;
        (r * a.cos(), r * a.sin())
    };
    let prism = |plan: &[(f64, f64)], z0: f64, z1: f64| -> Vec<Vec3> { [z0, z1].iter().flat_map(|&z| plan.iter().map(move |&(x, y)| Vec3::new(x, y, z))).collect() };
    let core_plan: Vec<(f64, f64)> = (0..n).map(|k| corner(k, L::CORE)).collect();
    let column_plan = |k: usize| -> Vec<(f64, f64)> {
        // Under every other outer sector, on its bisector.
        let a = 2.0 * std::f64::consts::PI * (2 * k) as f64 / n as f64 + std::f64::consts::PI / n as f64;
        let (cx, cy) = (L::COLUMN_AT * a.cos(), L::COLUMN_AT * a.sin());
        (0..8).map(|m| m as f64 * std::f64::consts::PI / 4.0 + std::f64::consts::PI / 8.0).map(|t| (cx + L::COLUMN * t.cos(), cy + L::COLUMN * t.sin())).collect()
    };
    let base = L::FOOTING_DEPTH;
    let top = |f: usize| base + (f + 1) as f64 * L::STOREY;
    let rc = "reinforced-concrete";
    let mut chunks = Vec::with_capacity(L::CHUNKS);
    chunks.push(footing(hull_chunk(&prism(&core_plan, 0.0, base), "footing-anchor", "footing", "footing".into())));
    for k in 0..L::COLUMNS {
        chunks.push(footing(hull_chunk(&prism(&column_plan(k), 0.0, base), "footing-anchor", "footing", "footing".into())));
    }
    for f in 0..L::FLOORS {
        let (z1, floor) = (top(f), format!("floor-{}", f + 1));
        let z0 = if f == 0 { base } else { top(f - 1) };
        chunks.push(hull_chunk(&prism(&core_plan, z0, z1), rc, "core", floor.clone()));
        for (r0, r1) in [(L::CORE, L::RING), (L::RING, L::EDGE)] {
            for k in 0..n {
                let plan = [corner(k, r0), corner(k + 1, r0), corner(k + 1, r1), corner(k, r1)];
                chunks.push(hull_chunk(&prism(&plan, z1 - L::SLAB, z1), rc, "slab", floor.clone()));
            }
        }
        for k in 0..L::COLUMNS {
            chunks.push(hull_chunk(&prism(&column_plan(k), z0, z1 - L::SLAB), rc, "column", floor.clone()));
        }
    }
    let bonds = between_free(&chunks, face_bonds(&chunks, joint_of));
    Structure {
        key: "wy_round_tower",
        title: "Round tower on columns",
        chunks,
        bonds,
        aim: {
            // The column at angle pi + pi / 16 (k = 4): its -x side.
            let a = std::f64::consts::PI * (1.0 + 1.0 / n as f64);
            Aim { target: [L::COLUMN_AT * a.cos() - L::COLUMN, L::COLUMN_AT * a.sin(), base + 0.5 * L::STOREY], direction: [1.0, 0.0, 0.0] }
        },
        ball: BALLS[0],
    }
}

/// The brick tower's dimensions (m).
pub struct BrickLayout;

impl BrickLayout {
    /// Outer plan side, wall thickness (one brick), brick length and course height.
    pub const SIDE: f64 = 4.0;
    pub const WALL: f64 = 0.25;
    pub const BRICK: f64 = 0.5;
    pub const COURSE: f64 = 0.25;
    pub const COURSES: usize = 48;
    pub const FOOTING_DEPTH: f64 = 0.5;
    /// Bricks per course: two full-length walls of 8, two between them of 7.
    pub const PER_COURSE: usize = 2 * 8 + 2 * 7;
    pub const CHUNKS: usize = Self::COURSES * Self::PER_COURSE + 4;
}

/// A 4 m square brick shaft 12 m tall: 48 courses of 0.5 x 0.25 x 0.25 m bricks in
/// running bond, the corners interlocked (the long walls alternate between courses, so
/// the joints of one course fall at the middle of the bricks below), mortar joints, on a
/// concrete strip footing. 1444 chunks. The ball comes in along +x at the middle of the
/// -x wall, a quarter of the way up.
pub fn brick_tower() -> Structure {
    type L = BrickLayout;
    let (s, t, b) = (0.5 * L::SIDE, L::WALL, L::BRICK);
    let base = L::FOOTING_DEPTH;
    let mut chunks = Vec::with_capacity(L::CHUNKS);
    // The strip footing: two full sides, two between them.
    for (lo, hi) in [([-s, -s], [s, -s + t]), ([-s, s - t], [s, s]), ([-s, -s + t], [-s + t, s - t]), ([s - t, -s + t], [s, s - t])] {
        chunks.push(footing(block([lo[0], lo[1], 0.0], [hi[0], hi[1], base], "footing-anchor", "footing", "footing".into())));
    }
    for c in 0..L::COURSES {
        let (z0, z1) = (base + c as f64 * L::COURSE, base + (c + 1) as f64 * L::COURSE);
        let group = format!("course-{}", c + 1);
        // Even courses: the y walls (front, back) run the full side; odd: the x walls.
        let full_along_x = c % 2 == 0;
        for wall in 0..4 {
            let along_x = (wall < 2) == full_along_x;
            let (a0, a1) = if wall < 2 { (-s, s) } else { (-s + t, s - t) };
            let side = if wall % 2 == 0 { -s } else { s - t };
            let count = ((a1 - a0) / b).round() as usize;
            for k in 0..count {
                let (u0, u1) = (a0 + k as f64 * b, a0 + (k + 1) as f64 * b);
                let (lo, hi) = if along_x { ([u0, side, z0], [u1, side + t, z1]) } else { ([side, u0, z0], [side + t, u1, z1]) };
                chunks.push(block(lo, hi, "brick", "brick", group.clone()));
            }
        }
    }
    let bonds = between_free(&chunks, auto_bonds(&chunks, joint_of));
    Structure {
        key: "wy_brick_tower",
        title: "Brick tower",
        chunks,
        bonds,
        aim: Aim { target: [-s, 0.0, base + 0.25 * L::COURSES as f64 * L::COURSE], direction: [1.0, 0.0, 0.0] },
        ball: BALLS[0],
    }
}
