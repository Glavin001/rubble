//! The authored showcase vehicles (`engine_softbody::authoring`) for pages: the
//! node–beam skeleton as `.truck` text for `WebWorld::explicitSim` (and every
//! other `.truck` consumer), and the visual skin bound to it, as typed arrays for
//! `showcase-car-render.mjs`.
//!
//! `params` (all optional): `{ style: 'sedan' | 'hatchback', resolution: 'low' |
//! 'medium' | 'high', strength (× every yield and break threshold, default 1),
//! paint: [r, g, b] (linear) }`.

use crate::err;
use engine_softbody::authoring::skin::{
    showcase_visual, showcase_visual_lod, ShowcaseVisual, MATERIAL_NAMES,
};
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use js_sys::{Array, Float32Array, Object, Reflect, Uint32Array, Uint8Array};
use wasm_bindgen::prelude::*;

fn built(params: &JsValue) -> Result<Built, JsValue> {
    let v: serde_json::Value = if params.is_undefined() || params.is_null() {
        serde_json::Value::Null
    } else {
        serde_wasm_bindgen::from_value(params.clone()).map_err(err)?
    };
    let name = v.get("style").and_then(|s| s.as_str()).unwrap_or("sedan");
    let def =
        style(name).ok_or_else(|| err(format!("unknown body style `{name}`: sedan, hatchback")))?;
    let res_name = v
        .get("resolution")
        .and_then(|s| s.as_str())
        .unwrap_or("medium");
    let resolution = Resolution::parse(res_name).ok_or_else(|| {
        err(format!(
            "unknown resolution `{res_name}`: low, medium, high"
        ))
    })?;
    let strength = v.get("strength").and_then(|s| s.as_f64()).unwrap_or(1.0) as f32;
    let paint = v.get("paint").and_then(|p| p.as_array()).and_then(|a| {
        (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.5) as f32))
    });
    build(
        &def,
        &BuildParams {
            resolution,
            strength,
            paint,
        },
    )
    .map_err(err)
}

fn set(o: &Object, k: &str, v: &JsValue) {
    let _ = Reflect::set(o, &JsValue::from_str(k), v);
}

fn f32s(v: impl IntoIterator<Item = f32>) -> JsValue {
    let v: Vec<f32> = v.into_iter().collect();
    Float32Array::from(v.as_slice()).into()
}

/// The showcase vehicle's `.truck` text (the explicit solver's input; Rigs of
/// Rods format). Its forward axis is +z, centred on its length.
#[wasm_bindgen(js_name = showcaseCarTruck)]
pub fn showcase_car_truck(params: JsValue) -> Result<String, JsValue> {
    Ok(built(&params)?.truck)
}

/// The showcase vehicle: `{ truck, counts: { nodes, beams, collisionTriangles,
/// skinTriangles, wheels, stiffnessScale, mass }, visual }` with `visual` =
/// `{ positions, normals, uvs, colors (Float32Array, 3 or 2 per vertex),
/// material (Uint8Array, a material id per vertex), materials (names by id),
/// indices (Uint32Array), bindNodes (Uint32Array, 3 solver node indices per
/// vertex, tyre nodes included), bindCoords (Float32Array, α β h per vertex),
/// bindNormal (Float32Array, the rest normal in the triangle frame), bindRest
/// (Float32Array, the triangle's rest |e₁| |e₂|), ao (Float32Array, baked
/// ambient occlusion, 1 open), smooth (Uint8Array), rigid (Uint8Array: 1 where
/// (α, β, h) are an offset in the triangle's orthonormal frame), the damage
/// units (unitNames, unitStart, unitBeams, unitKind, unitStrain, unitShards,
/// unit and shard per vertex, beamRest0, nodeBeamStart, nodeBeamList,
/// nodeMaterial: `visual_js`), nodeRest
/// (Float32Array, every solver node's rest position: the renderer's tears), normalTris, nodeTriStart, nodeTriList
/// (Uint32Array), nodeRestNormal (Float32Array) (smooth shading from node
/// normals; see `ShowcaseVisual::smooth`), nodeCount, vertexCount }` and
/// `visualLod`, the same at a distant level of detail (about a third of the
/// triangles; `showcase-car-render.mjs` switches to it by distance).
/// `showcase-car-render.mjs` documents the binding.
#[wasm_bindgen(js_name = showcaseCar)]
pub fn showcase_car(params: JsValue) -> Result<JsValue, JsValue> {
    let b = built(&params)?;
    let v = showcase_visual(&b).map_err(err)?;
    let out = Object::new();
    set(&out, "truck", &JsValue::from_str(&b.truck));
    let counts = Object::new();
    let sk = &b.skeleton;
    set(&counts, "nodes", &JsValue::from(sk.nodes.len() as u32));
    set(&counts, "beams", &JsValue::from(sk.beams.len() as u32));
    set(
        &counts,
        "collisionTriangles",
        &JsValue::from(sk.collision.len() as u32),
    );
    set(
        &counts,
        "skinTriangles",
        &JsValue::from(sk.skin.len() as u32),
    );
    set(&counts, "wheels", &JsValue::from(sk.wheels.len() as u32));
    set(&counts, "stiffnessScale", &JsValue::from(b.stiffness_scale));
    set(&counts, "mass", &JsValue::from(sk.total_mass()));
    set(&out, "counts", &counts);
    set(&out, "visual", &visual_js(&v).into());
    // The distant level of detail (the renderer picks it by distance).
    let lod = showcase_visual_lod(&b, 0.45).map_err(err)?;
    set(&out, "visualLod", &visual_js(&lod).into());
    Ok(out.into())
}

/// The visual alone (`showcaseCar(params).visual`).
#[wasm_bindgen(js_name = showcaseCarVisual)]
pub fn showcase_car_visual(params: JsValue) -> Result<JsValue, JsValue> {
    let car = showcase_car(params)?;
    Reflect::get(&car, &JsValue::from_str("visual"))
}

fn visual_js(v: &ShowcaseVisual) -> Object {
    let vis = Object::new();
    set(
        &vis,
        "positions",
        &f32s(v.positions.iter().flatten().copied()),
    );
    set(&vis, "normals", &f32s(v.normals.iter().flatten().copied()));
    set(&vis, "uvs", &f32s(v.uvs.iter().flatten().copied()));
    set(&vis, "colors", &f32s(v.colors.iter().flatten().copied()));
    set(
        &vis,
        "material",
        &Uint8Array::from(v.material.as_slice()).into(),
    );
    let names = Array::new();
    for n in MATERIAL_NAMES {
        names.push(&JsValue::from_str(n));
    }
    set(&vis, "materials", &names.into());
    set(
        &vis,
        "indices",
        &Uint32Array::from(v.indices.as_slice()).into(),
    );
    let nodes: Vec<u32> = v.bind_nodes.iter().flatten().copied().collect();
    set(
        &vis,
        "bindNodes",
        &Uint32Array::from(nodes.as_slice()).into(),
    );
    set(
        &vis,
        "bindCoords",
        &f32s(v.bind_coords.iter().flatten().copied()),
    );
    set(
        &vis,
        "bindNormal",
        &f32s(v.bind_normal.iter().flatten().copied()),
    );
    set(
        &vis,
        "bindRest",
        &f32s(v.bind_rest.iter().flatten().copied()),
    );
    set(&vis, "ao", &f32s(v.ao.iter().copied()));
    set(
        &vis,
        "smooth",
        &Uint8Array::from(v.smooth.as_slice()).into(),
    );
    set(&vis, "rigid", &Uint8Array::from(v.rigid.as_slice()).into());
    // Damage units (skin.rs `DamageUnit`): per unit its name, kind, beams
    // (CSR: unitStart, unitBeams, this vehicle's solver beam indices), strain
    // trigger and shard count; per vertex its unit (u32 max: none) and shard
    // seed; the solver beams' spawn rest lengths; per node its beams (CSR,
    // tyre beams excluded) and surface material.
    let names = Array::new();
    let mut start = vec![0u32];
    let mut ubeams = Vec::new();
    for u in &v.units {
        names.push(&JsValue::from_str(&u.name));
        ubeams.extend_from_slice(&u.beams);
        start.push(ubeams.len() as u32);
    }
    set(&vis, "unitNames", &names.into());
    set(&vis, "unitStart", &Uint32Array::from(start.as_slice()).into());
    set(&vis, "unitBeams", &Uint32Array::from(ubeams.as_slice()).into());
    let kinds: Vec<u8> = v.units.iter().map(|u| u.kind).collect();
    set(&vis, "unitKind", &Uint8Array::from(kinds.as_slice()).into());
    set(&vis, "unitStrain", &f32s(v.units.iter().map(|u| u.strain.min(f32::MAX))));
    let shards: Vec<u32> = v.units.iter().map(|u| u.shards).collect();
    set(&vis, "unitShards", &Uint32Array::from(shards.as_slice()).into());
    set(&vis, "unit", &Uint32Array::from(v.unit.as_slice()).into());
    set(&vis, "shard", &Uint8Array::from(v.shard.as_slice()).into());
    set(&vis, "beamRest0", &f32s(v.beam_rest0.iter().copied()));
    set(
        &vis,
        "nodeBeamStart",
        &Uint32Array::from(v.node_beam_start.as_slice()).into(),
    );
    set(
        &vis,
        "nodeBeamList",
        &Uint32Array::from(v.node_beam_list.as_slice()).into(),
    );
    set(
        &vis,
        "nodeMaterial",
        &Uint8Array::from(v.node_material.as_slice()).into(),
    );
    set(
        &vis,
        "nodeRest",
        &f32s(v.node_rest.iter().flatten().copied()),
    );
    let nt: Vec<u32> = v.normal_tris.iter().flatten().copied().collect();
    set(&vis, "normalTris", &Uint32Array::from(nt.as_slice()).into());
    set(
        &vis,
        "nodeTriStart",
        &Uint32Array::from(v.node_tri_start.as_slice()).into(),
    );
    set(
        &vis,
        "nodeTriList",
        &Uint32Array::from(v.node_tri_list.as_slice()).into(),
    );
    set(
        &vis,
        "nodeRestNormal",
        &f32s(v.node_rest_normal.iter().flatten().copied()),
    );
    set(&vis, "nodeCount", &JsValue::from(v.node_count));
    set(
        &vis,
        "vertexCount",
        &JsValue::from(v.positions.len() as u32),
    );
    vis
}

/// The visual layer of an imported BeamNG JBeam vehicle (engine_softbody
/// `flexmesh`): its flexbodies and props, bound to the explicit solver's nodes
/// as `explicitSim` spawns it from `jbeam` (a desc node maps to the solver node
/// at its rest position), in `showcaseCar`'s `visual` format, so
/// `showcase-car-render.mjs` draws it. `spec`: `{ jbeam: [[name, text]], pc,
/// paint: [r, g, b], native (the JBeam-mode spawn: `explicitSim`'s),
/// materials: [[name, text]] (the `*.materials.json` files) }`; `meshes`: `[[name, Uint8Array]]`, the vehicle's `.dae`
/// files. Returns `{ visual, visualLod (the distant level: the meshes
/// clustered on a 3 cm grid), lodDistance (m: where a cluster is about 2.6
/// pixels), lodTriangles, report: { flexbodies, props, vertices, triangles,
/// unbound, missing } }`.
#[wasm_bindgen(js_name = jbeamVisual)]
pub fn jbeam_visual(spec: JsValue, meshes: Array) -> Result<JsValue, JsValue> {
    use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
    use engine_softbody::flexmesh;
    use glam::Vec3;
    let v: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
    let files: Vec<(String, String)> = v
        .get("jbeam")
        .and_then(|f| f.as_array())
        .ok_or_else(|| err("jbeamVisual: `jbeam` files"))?
        .iter()
        .map(|f| {
            (
                f.get(0).and_then(|x| x.as_str()).unwrap_or("").to_string(),
                f.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string(),
            )
        })
        .collect();
    let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let pc = v.get("pc").and_then(|t| t.as_str());
    let paint = v
        .get("paint")
        .and_then(|p| p.as_array())
        .filter(|a| a.len() == 3)
        .map(|a| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.5) as f32))
        .unwrap_or([0.45, 0.05, 0.04]);
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pc).map_err(err)?;
    let flex = engine_softbody::jbeam::assembled_rows(&refs, pc, "flexbodies").map_err(err)?;
    let props = engine_softbody::jbeam::assembled_rows(&refs, pc, "props").map_err(err)?;
    // The solver path the lab spawns it on: `native` the JBeam mode (desc
    // order, pressure wheels), else the `.truck` path. Either way the visual
    // binds by rest position, so this only has to match the spawn's nodes.
    let sim = if v.get("native").and_then(|x| x.as_bool()).unwrap_or(false) {
        ExplicitBeams::from_desc_jbeam_with(
            &desc,
            engine_softbody::explicit_beams::JbeamWheels::Pressure,
            Vec3::ZERO,
            Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
        .map_err(err)?
        .0
    } else {
        ExplicitBeams::from_jbeam(&refs, pc, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default())
            .map_err(err)?
            .0
    };
    let mut all = Vec::new();
    for m in meshes.iter() {
        let pair = Array::from(&m);
        let name = pair.get(0).as_string().unwrap_or_default();
        let bytes = Uint8Array::new(&pair.get(1)).to_vec();
        let text = String::from_utf8_lossy(&bytes);
        all.extend(flexmesh::read_collada(&text).map_err(|e| err(format!("{name}: {e}")))?);
    }
    let mut mats = std::collections::HashMap::new();
    for m in v.get("materials").and_then(|m| m.as_array()).into_iter().flatten() {
        mats.extend(flexmesh::read_materials(m.get(1).and_then(|x| x.as_str()).unwrap_or("")));
    }
    let (vis, rep) = flexmesh::jbeam_visual(&flex, &props, &desc, &sim, &all, &mats, paint).map_err(err)?;
    // The distant level: the meshes clustered (flexmesh::cluster).
    let far: Vec<flexmesh::DaeMesh> = all.iter().map(|m| flexmesh::cluster(m, flexmesh::LOD_CELL)).collect();
    let (lod, lrep) = flexmesh::jbeam_visual(&flex, &props, &desc, &sim, &far, &mats, paint).map_err(err)?;
    let out = Object::new();
    set(&out, "visual", &visual_js(&vis).into());
    set(&out, "visualLod", &visual_js(&lod).into());
    set(&out, "lodDistance", &JsValue::from(flexmesh::lod_distance(flexmesh::LOD_CELL)));
    set(&out, "lodTriangles", &JsValue::from(lrep.triangles as u32));
    let r = Object::new();
    set(&r, "flexbodies", &JsValue::from(rep.flexbodies as u32));
    set(&r, "props", &JsValue::from(rep.props as u32));
    set(&r, "vertices", &JsValue::from(rep.vertices as u32));
    set(&r, "triangles", &JsValue::from(rep.triangles as u32));
    set(&r, "unbound", &JsValue::from(rep.unbound as u32));
    let missing = Array::new();
    for m in &rep.missing {
        missing.push(&JsValue::from_str(m));
    }
    set(&r, "missing", &missing.into());
    set(&out, "report", &r.into());
    Ok(out.into())
}
