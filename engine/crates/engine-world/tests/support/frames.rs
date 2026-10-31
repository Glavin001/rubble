//! The support-loss test frames (`tests/support_loss.rs`): a slab or block on columns on
//! footings, built as a scene pack with the bond strengths and masses set exactly. Shared
//! with `examples/reaction_replay.rs` (`#[path]`), so it stays out of the public API.
#![allow(dead_code)]

use serde_json::{json, Value};

pub const G: f64 = 9.81;
/// Column section (m): 0.4 x 0.4, so every bond's area is 0.16 m².
pub const COLUMN_HALF: f64 = 0.2;
pub const AREA: f64 = 4.0 * COLUMN_HALF * COLUMN_HALF;

/// A pack: `columns` at the (x, z) given on footings (mass 0: supports), carrying one `top`
/// chunk of `top_mass` kg and half extents `top_half`. Column bonds hold `compression` Pa;
/// the columns listed in `weak` are bonded with negligible tension and shear (the shot breaks
/// them), the rest with plenty. Returns the pack and its chunk indices.
pub struct Frame {
    pub pack: Vec<u8>,
    /// Chunk indices: the footings, the columns (both in `columns` order) and the top.
    pub footings: Vec<u32>,
    pub columns: Vec<u32>,
    pub top: u32,
    /// The top's half extents.
    pub top_half: [f64; 3],
}

pub fn frame(
    columns: &[[f64; 2]],
    weak: &[usize],
    top_mass: f64,
    top_half: [f64; 3],
    compression: f64,
) -> Frame {
    let (footing_h, column_h) = (0.25, 1.0);
    let mut nodes = Vec::new();
    let mut sizes = Vec::new();
    let mut colliders = Vec::new();
    let mut materials = Vec::new();
    let mut node = |centre: [f64; 3], half: [f64; 3], mass: f64| {
        nodes.push(
            json!({ "centroid": { "x": centre[0], "y": centre[1], "z": centre[2] }, "mass": mass }),
        );
        sizes.push(json!({ "x": 2.0 * half[0], "y": 2.0 * half[1], "z": 2.0 * half[2] }));
        colliders.push(json!({ "kind": "cuboid", "halfExtents": { "x": half[0], "y": half[1], "z": half[2] } }));
        materials.push("concrete");
        nodes.len() as u32 - 1
    };
    let footings: Vec<u32> = columns
        .iter()
        .map(|&[x, z]| node([x, footing_h, z], [0.3, footing_h, 0.3], 0.0))
        .collect();
    let column_y = 2.0 * footing_h + column_h;
    let ids: Vec<u32> = columns
        .iter()
        .map(|&[x, z]| {
            node(
                [x, column_y, z],
                [COLUMN_HALF, column_h, COLUMN_HALF],
                400.0,
            )
        })
        .collect();
    let deck = 2.0 * (footing_h + column_h);
    let top = node([0.0, deck + top_half[1], 0.0], top_half, top_mass);
    let mut bonds = Vec::new();
    for (k, &[x, z]) in columns.iter().enumerate() {
        let m = if weak.contains(&k) { 1 } else { 0 };
        for (a, b, y) in [(footings[k], ids[k], 2.0 * footing_h), (ids[k], top, deck)] {
            bonds.push(json!({
                "node0": a, "node1": b, "m": m, "area": AREA,
                "centroid": { "x": x, "y": y, "z": z }, "normal": { "x": 0, "y": 1, "z": 0 },
            }));
        }
    }
    let material = |name: &str, tension: f64| {
        json!({
            "name": name, "elasticModulus": 30e9,
            "compressionElastic": compression, "compressionFatal": compression,
            "tensionElastic": tension, "tensionFatal": tension,
            "shearElastic": tension, "shearFatal": tension,
        })
    };
    let pack: Value = json!({
        "version": 2, "key": "support_frame", "title": "Support-loss frame",
        "defaults": { "solver": { "gravity": -G, "materialScale": 1,
            "materials": [material("concrete-member", 1e10), material("concrete-shot", 1e3)] } },
        "scenario": {
            "nodes": nodes, "nodeSizes": sizes, "nodeColliders": colliders,
            "nodeMaterials": materials, "bonds": bonds,
        },
    });
    Frame {
        pack: serde_json::to_vec(&pack).unwrap(),
        footings,
        columns: ids,
        top,
        top_half,
    }
}
