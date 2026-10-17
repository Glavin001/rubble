//! Shardwright `.fracphys` -> compact JSON for stress-ref (`shardwright-stress/1`):
//! fragments of levels 0..=max_level (id, level, parent, component, role, mass, com,
//! inertia, dominant material, hull vertices, and each hull's points and polygon faces) and bonds of one level (section data,
//! joint kind, rebar, strength scale). Asset frame, +Y up, SI units.
//!
//!   shard-export <asset.fracphys> <out.json> [bond_level=2] [asset.json]
//!
//! The `.fracphys` carries component ids only; with the bake's `.asset.json` the
//! components' names (e.g. `f0_col_21`) are written as `component_names`.
use frac_io::fb;
use serde_json::{json, Value};

fn v3(v: Option<&fb::Vec3>) -> Value {
    let v = v.expect("vec3");
    json!([v.x(), v.y(), v.z()])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bond_level: u8 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2);
    let bytes = std::fs::read(&args[1]).unwrap();
    let opts = flatbuffers::VerifierOptions { max_depth: 64, max_tables: usize::MAX / 2, max_apparent_size: 1 << 31, ..Default::default() };
    let a = fb::root_as_asset_with_opts(&opts, &bytes).expect("fracphys");
    let lib = frac_material::MaterialLibrary::builtin();
    let materials: Vec<String> = lib.materials.iter().map(|m| m.id.clone()).collect();
    let hulls = a.hulls().unwrap();
    let hull_faces = a.hull_faces().unwrap();
    let mut frags = Vec::new();
    for f in a.fragments().unwrap().iter() {
        if f.level() > bond_level {
            continue;
        }
        let ids: Vec<u16> = f.material_ids().map(|v| v.iter().collect()).unwrap_or_default();
        let fr: Vec<f32> = f.material_fracs().map(|v| v.iter().collect()).unwrap_or_default();
        let dom = ids.iter().zip(fr.iter()).max_by(|x, y| x.1.total_cmp(y.1)).map(|x| *x.0 as usize);
        let mut pts = Vec::new();
        let mut parts = Vec::new();
        for h in f.hull_begin()..f.hull_begin() + f.hull_count() as u32 {
            let part: Vec<Value> = hulls.get(h as usize).vertices().unwrap().iter().map(|p| json!([p.x(), p.y(), p.z()])).collect();
            // Its polygon faces (vertex indices, counter-clockwise seen from outside).
            let hf = hull_faces.get(h as usize);
            let indices: Vec<u32> = hf.indices().unwrap().iter().collect();
            let mut faces = Vec::new();
            let mut at = 0usize;
            for size in hf.face_sizes().unwrap().iter() {
                faces.push(json!(indices[at..at + size as usize]));
                at += size as usize;
            }
            pts.extend(part.iter().cloned());
            parts.push(json!({ "points": part, "faces": faces }));
        }
        let m = f.inertia().unwrap().m();
        frags.push(json!({
            "id": f.id(), "level": f.level(), "parent": f.parent(), "component": f.component(), "role": f.role(),
            "mass": f.mass(), "com": v3(f.com()), "inertia": (0..9).map(|k| m.get(k)).collect::<Vec<f32>>(),
            "volume": f.volume(), "material": dom.and_then(|d| materials.get(d).cloned()), "hull_count": f.hull_count(), "points": pts, "hulls": parts
        }));
    }
    let mut bonds = Vec::new();
    for b in a.bonds().unwrap().iter() {
        if b.level() != bond_level {
            continue;
        }
        let comp: Vec<Value> = b.composition().map(|c| c.iter().map(|k| json!([k.kind(), k.interface_material(), k.fraction()])).collect()).unwrap_or_default();
        let eh = b.extent_half().unwrap();
        bonds.push(json!({
            "id": b.id(), "a": b.a(), "b": b.b(), "area": b.area(), "centroid": v3(b.centroid()), "normal": v3(b.normal()),
            "frame_u": v3(b.frame_u()), "frame_v": v3(b.frame_v()), "i_uu": b.i_uu(), "i_vv": b.i_vv(), "i_uv": b.i_uv(), "j": b.j(),
            "extent_half": [eh.x(), eh.y()], "dist_a": b.dist_a(), "dist_b": b.dist_b(), "composition": comp,
            "rebar_count": b.rebar_count(), "rebar_area": b.rebar_area(), "strength_scale": b.strength_scale(), "parent_bond": b.parent_bond()
        }));
    }
    let component_names: Vec<Value> = args.get(4).map(|p| {
        let asset: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).expect("asset.json");
        asset["components"].as_array().expect("components").iter().map(|c| json!([c["id"], c["name"]])).collect()
    }).unwrap_or_default();
    let out = json!({ "format": "shardwright-stress/1", "name": a.name(), "materials": materials, "bond_level": bond_level, "fragments": frags, "bonds": bonds, "component_names": component_names });
    std::fs::write(&args[2], serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("{} fragments (levels 0..={bond_level}), {} bonds at level {bond_level}", frags.len(), bonds.len());
}
