//! A fracture GLB as the stress oracle's Shardwright input (`shardwright-stress/1`, read
//! by `stress_ref::shardwright`): fragments of levels `0..=level` with their hulls, and the
//! bonds of `level`. The same JSON as the Shardwright repository's `shard-export` tool
//! writes from a `.fracphys`, so the oracle runs exactly the asset the engine loads.
//! Asset frame (+Y up), SI units.

use serde_json::{json, Value};

use crate::asset::FractureAsset;

/// The material ids of the asset's embedded library, by index.
pub fn material_names(asset: &FractureAsset) -> Vec<String> {
    asset.json["extensions"][crate::asset::EXTENSION]["materialLibrary"]["material"]
        .as_array()
        .map(|m| m.iter().map(|m| m["id"].as_str().unwrap_or("").to_string()).collect())
        .unwrap_or_default()
}

pub fn shardwright_stress_json(asset: &FractureAsset, level: u8) -> Value {
    let p = &asset.physics;
    let materials = material_names(asset);
    let mut fragments = Vec::new();
    for f in p.fragments.iter().filter(|f| f.level <= level) {
        let dominant = f
            .material_ids
            .iter()
            .zip(&f.material_fractions)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .and_then(|(id, _)| materials.get(*id as usize).cloned());
        let mut points = Vec::new();
        let mut hulls = Vec::new();
        let hulls_range = f.hull_begin as usize..(f.hull_begin + f.hull_count) as usize;
        for (h, topo) in p.hulls[hulls_range.clone()].iter().zip(&p.hull_faces[hulls_range]) {
            let mut faces = Vec::new();
            let mut at = 0usize;
            for &size in &topo.sizes {
                faces.push(json!(topo.indices[at..at + size as usize]));
                at += size as usize;
            }
            points.extend(h.vertices.iter().cloned());
            hulls.push(json!({ "points": h.vertices, "faces": faces }));
        }
        fragments.push(json!({
            "id": f.id, "level": f.level, "parent": f.parent, "component": f.component, "role": f.role,
            "mass": f.mass, "com": f.com, "inertia": f.inertia, "volume": f.volume,
            "material": dominant, "hull_count": f.hull_count, "points": points, "hulls": hulls,
        }));
    }
    let bonds: Vec<Value> = p
        .bonds
        .iter()
        .filter(|b| b.level == level)
        .map(|b| {
            let composition: Vec<Value> = b.composition.iter().map(|c| json!([c.kind, c.interface_material, c.fraction])).collect();
            json!({
                "id": b.id, "a": b.a, "b": b.b, "area": b.area, "centroid": b.centroid, "normal": b.normal,
                "frame_u": b.frame_u, "frame_v": b.frame_v, "i_uu": b.i_uu, "i_vv": b.i_vv, "i_uv": b.i_uv, "j": b.polar_moment,
                "extent_half": b.extent_half, "dist_a": b.dist_a, "dist_b": b.dist_b, "composition": composition,
                "rebar_count": b.rebar_count, "rebar_area": b.rebar_area, "strength_scale": b.strength_scale, "parent_bond": b.parent_bond,
            })
        })
        .collect();
    json!({
        "format": "shardwright-stress/1", "name": p.name, "materials": materials, "bond_level": level,
        "fragments": fragments, "bonds": bonds, "component_names": [],
    })
}
