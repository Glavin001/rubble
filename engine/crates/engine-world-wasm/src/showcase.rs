//! The authored showcase vehicles (`engine_softbody::authoring`) for pages: the
//! node–beam skeleton as `.truck` text for `WebWorld::explicitSim` (and every
//! other `.truck` consumer), and the visual skin bound to it, as typed arrays for
//! `showcase-car-render.mjs`.
//!
//! `params` (all optional): `{ style: 'sedan' | 'hatchback', resolution: 'low' |
//! 'medium' | 'high', strength (× every yield and break threshold, default 1),
//! paint: [r, g, b] (linear) }`.

use crate::err;
use engine_softbody::authoring::skin::{showcase_visual_lod, ShowcaseVisual, MATERIAL_NAMES};
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
/// A texture file (PNG or DDS) as one renderer layer: `size` × `size` RGBA8
/// with its mips, concatenated (`engine_softbody::flextex`).
#[wasm_bindgen(js_name = decodeTexture)]
pub fn decode_texture(bytes: &[u8], size: u32) -> Result<Vec<u8>, JsValue> {
    let img = engine_softbody::flextex::decode(bytes).map_err(err)?;
    Ok(engine_softbody::flextex::layer(&img, size))
}

/// The showcase car's third level of detail (`showcase_visual_lod`'s detail).
const FAR_DETAIL: f32 = 0.2;

/// The visual for a spawn on raycast wheels (`engine_softbody::explicit_drive`):
/// without the tyre nodes and beams the `.truck` spawn builds, so every binding,
/// beam index and per-node array matches that solver's vehicle. The wheels'
/// pieces (tyre, rim, disc, caliper) ride the raycast wheels instead
/// (`skin::bind_raycast_wheels`: the renderer places them from the drive
/// state); beams after the tyres' move down.
struct TyreCut {
    /// The spawn's nodes without tyres, its first tyre beam and the tyre beams.
    nodes: u32,
    first: u32,
    count: u32,
    /// The raycast spawn's drive (its wheels, in the drive state's order).
    config: engine_softbody::explicit_drive::DriveConfig,
}

impl TyreCut {
    fn of(truck: &str) -> Result<Self, JsValue> {
        use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
        let z = glam::Vec3::ZERO;
        let with = ExplicitBeams::from_truck(truck, z, z, ExplicitBeamsSettings::default())
            .map_err(err)?;
        let (without, _) = ExplicitBeams::from_truck_raycast(
            truck,
            None,
            z,
            0.0,
            z,
            ExplicitBeamsSettings::default(),
        )
        .map_err(err)?;
        // Beam order: cinecam beams, tyre beams, then the file's.
        let first = engine_softbody::truck::parse_truck(truck).cinecams.len() as u32 * 8;
        let config = without
            .drive
            .as_ref()
            .ok_or_else(|| err("raycast spawn without a drive"))?
            .config
            .clone();
        Ok(Self {
            nodes: without.nodes.len() as u32,
            first,
            count: (with.beams.len() - without.beams.len()) as u32,
            config,
        })
    }

    fn beam(&self, b: u32) -> Option<u32> {
        if b < self.first {
            Some(b)
        } else if b < self.first + self.count {
            None
        } else {
            Some(b - self.count)
        }
    }

    fn apply(&self, v: &mut ShowcaseVisual) -> Result<(), JsValue> {
        engine_softbody::authoring::skin::bind_raycast_wheels(v, &self.config).map_err(err)?;
        let n = self.nodes;
        let live = |t: &[u32; 3]| t.iter().all(|&i| i < n);
        let ok_vertex: Vec<bool> = v.bind_nodes.iter().map(live).collect();
        let idx: Vec<u32> = v
            .indices
            .chunks(3)
            .filter(|t| t.iter().all(|&i| ok_vertex[i as usize]))
            .flatten()
            .copied()
            .collect();
        v.indices = idx;
        for (b, ok) in v.bind_nodes.iter_mut().zip(&ok_vertex) {
            if !ok {
                *b = [0; 3];
            }
        }
        v.normal_tris.retain(|t| live(t));
        let mut lists = vec![Vec::new(); n as usize];
        for (k, t) in v.normal_tris.iter().enumerate() {
            for &i in t {
                lists[i as usize].push(k as u32);
            }
        }
        v.node_tri_start = std::iter::once(0)
            .chain(lists.iter().scan(0u32, |a, l| {
                *a += l.len() as u32;
                Some(*a)
            }))
            .collect();
        v.node_tri_list = lists.concat();
        v.node_rest_normal.truncate(n as usize);
        v.node_rest.truncate(n as usize);
        v.node_material.truncate(n as usize);
        v.node_beam_start.truncate(n as usize + 1);
        let end = v.node_beam_start.last().copied().unwrap_or(0) as usize;
        v.node_beam_list.truncate(end);
        v.node_beam_list = v
            .node_beam_list
            .iter()
            .filter_map(|&b| self.beam(b))
            .collect();
        let (a, c) = (self.first as usize, (self.first + self.count) as usize);
        if c <= v.beam_rest0.len() {
            v.beam_rest0.drain(a..c);
        }
        for u in &mut v.units {
            u.beams = u.beams.iter().filter_map(|&b| self.beam(b)).collect();
        }
        // The shut-line fallbacks' death lists name solver beams: renumber
        // them past the cut tyre beams (a tyre beam never kills a skin frame).
        for d in &mut v.skin_death {
            for b in d.iter_mut() {
                if *b != u32::MAX {
                    *b = self.beam(*b).unwrap_or(u32::MAX);
                }
            }
        }
        v.node_count = n;
        Ok(())
    }
}

#[wasm_bindgen(js_name = showcaseCar)]
pub fn showcase_car(params: JsValue) -> Result<JsValue, JsValue> {
    let b = built(&params)?;
    // `wheels: 'raycast'`: the visual for a spawn on raycast wheels (no tyres).
    let raycast = !params.is_undefined()
        && !params.is_null()
        && Reflect::get(&params, &JsValue::from_str("wheels"))
            .ok()
            .and_then(|w| w.as_string())
            .as_deref()
            == Some("raycast");
    let cut = if raycast {
        Some(TyreCut::of(&b.truck)?)
    } else {
        None
    };
    let fix = |mut v: ShowcaseVisual| -> Result<ShowcaseVisual, JsValue> {
        if let Some(c) = &cut {
            c.apply(&mut v)?;
        }
        Ok(v)
    };
    // Measurement switches (a visual A/B on identical physics): `bind:
    // 'legacy'` the binding before the flexbody rebinding; `alt: false` no
    // shut-line fallbacks.
    let p: serde_json::Value = serde_wasm_bindgen::from_value(params.clone()).unwrap_or_default();
    let legacy = p.get("bind").and_then(|s| s.as_str()) == Some("legacy");
    let no_alt = p.get("alt").and_then(|s| s.as_bool()) == Some(false);
    let make = |detail: f32| {
        let v = if legacy {
            engine_softbody::authoring::skin::showcase_visual_legacy_lod(&b, detail)
        } else {
            showcase_visual_lod(&b, detail)
        };
        v.map(|mut v| {
            if no_alt {
                v.alt.clear();
            }
            v
        })
        .map_err(err)
    };
    let v = fix(make(1.0)?)?;
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
    let lod = fix(make(0.45)?)?;
    set(&out, "visualLod", &visual_js(&lod).into());
    // A third, coarser level (a car a few hundred pixels long and less).
    let far = fix(make(FAR_DETAIL)?)?;
    set(&out, "visualFar", &visual_js(&far).into());
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
    // Textures (imported vehicles' maps, skin.rs `ShowcaseVisual::tex_sets`):
    // colour and data file names, per vertex its texture set + 1 (0
    // untextured), per set its [colour, normal, roughness, AO] file + 1 (0
    // none; colour into `textures`, the rest into `dataTextures`).
    let names = |list: &[String]| {
        let a = Array::new();
        for t in list {
            a.push(&JsValue::from_str(t));
        }
        a
    };
    set(&vis, "textures", &names(&v.textures).into());
    set(&vis, "dataTextures", &names(&v.data_textures).into());
    let sets: Vec<u16> = v
        .tex_sets
        .iter()
        .flatten()
        .map(|&k| if k == u16::MAX { 0 } else { k + 1 })
        .collect();
    set(
        &vis,
        "texSets",
        &js_sys::Uint16Array::from(sets.as_slice()).into(),
    );
    if !v.tex.is_empty() {
        let t: Vec<u16> = v
            .tex
            .iter()
            .map(|&x| if x == u16::MAX { 0 } else { x + 1 })
            .collect();
        set(&vis, "tex", &js_sys::Uint16Array::from(t.as_slice()).into());
    }
    set(
        &vis,
        "edgeLength",
        &JsValue::from(engine_softbody::authoring::skin::median_edge(v)),
    );
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
    // Shut-line fallbacks (skin.rs `ShowcaseVisual::alt`): per vertex its
    // own-part binding + 1 (0 none), the bindings, the frame each replaces,
    // and per such frame the beams whose break kills it.
    set(&vis, "alt", &Uint32Array::from(v.alt.as_slice()).into());
    let an: Vec<u32> = v.alt_nodes.iter().flatten().copied().collect();
    set(&vis, "altNodes", &Uint32Array::from(an.as_slice()).into());
    set(
        &vis,
        "altCoords",
        &f32s(v.alt_coords.iter().flatten().copied()),
    );
    set(
        &vis,
        "altNormal",
        &f32s(v.alt_normal.iter().flatten().copied()),
    );
    set(&vis, "altRest", &f32s(v.alt_rest.iter().flatten().copied()));
    set(
        &vis,
        "altTri",
        &Uint32Array::from(v.alt_tri.as_slice()).into(),
    );
    let sd: Vec<u32> = v.skin_death.iter().flatten().copied().collect();
    set(&vis, "skinDeath", &Uint32Array::from(sd.as_slice()).into());
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
    set(
        &vis,
        "unitStart",
        &Uint32Array::from(start.as_slice()).into(),
    );
    set(
        &vis,
        "unitBeams",
        &Uint32Array::from(ubeams.as_slice()).into(),
    );
    let kinds: Vec<u8> = v.units.iter().map(|u| u.kind).collect();
    set(&vis, "unitKind", &Uint8Array::from(kinds.as_slice()).into());
    set(
        &vis,
        "unitStrain",
        &f32s(v.units.iter().map(|u| u.strain.min(f32::MAX))),
    );
    let shards: Vec<u32> = v.units.iter().map(|u| u.shards).collect();
    set(
        &vis,
        "unitShards",
        &Uint32Array::from(shards.as_slice()).into(),
    );
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
/// paint: [r, g, b], native (the JBeam-mode spawn: `explicitSim`'s), wheels
/// ('raycast': the drive scenes' spawn, no wheel nodes),
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
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
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
    // `wheels: 'raycast'` (the drive scenes' spawn): no wheel nodes.
    let raycast = v.get("wheels").and_then(|w| w.as_str()) == Some("raycast");
    let sim = if raycast && !v.get("native").and_then(|x| x.as_bool()).unwrap_or(false) {
        let (text, _) = engine_softbody::truck_export::desc_to_truck(
            &desc,
            engine_softbody::truck_export::TyreDefaults::for_vehicle(&desc),
        );
        ExplicitBeams::from_truck_raycast(&text, None, Vec3::ZERO, 0.0, Vec3::ZERO, ExplicitBeamsSettings::default())
            .map_err(err)?
            .0
    } else if v.get("native").and_then(|x| x.as_bool()).unwrap_or(false) {
        ExplicitBeams::from_desc_jbeam_with(
            &desc,
            if raycast {
                engine_softbody::explicit_beams::JbeamWheels::Raycast
            } else {
                engine_softbody::explicit_beams::JbeamWheels::Pressure
            },
            Vec3::ZERO,
            Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
        .map_err(err)?
        .0
    } else {
        ExplicitBeams::from_jbeam(
            &refs,
            pc,
            Vec3::ZERO,
            Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
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
    for m in v
        .get("materials")
        .and_then(|m| m.as_array())
        .into_iter()
        .flatten()
    {
        mats.extend(flexmesh::read_materials(
            m.get(1).and_then(|x| x.as_str()).unwrap_or(""),
        ));
    }
    let (vis, rep) =
        flexmesh::jbeam_visual(&flex, &props, &desc, &sim, &all, &mats, paint).map_err(err)?;
    // The distant level: the meshes clustered (flexmesh::cluster).
    let far: Vec<flexmesh::DaeMesh> = all
        .iter()
        .map(|m| flexmesh::cluster(m, flexmesh::LOD_CELL))
        .collect();
    let (lod, lrep) =
        flexmesh::jbeam_visual(&flex, &props, &desc, &sim, &far, &mats, paint).map_err(err)?;
    // A third, coarser level: clustered on FAR_CELL.
    let far_cell = v
        .get("farCell")
        .and_then(|x| x.as_f64())
        .map_or(flexmesh::FAR_CELL, |x| x as f32);
    let farther: Vec<flexmesh::DaeMesh> =
        all.iter().map(|m| flexmesh::cluster(m, far_cell)).collect();
    let (far_vis, _) =
        flexmesh::jbeam_visual(&flex, &props, &desc, &sim, &farther, &mats, paint).map_err(err)?;
    // Every level on the full level's texture layers.
    let (mut lod, mut far_vis) = (lod, far_vis);
    for l in [&mut lod, &mut far_vis] {
        flexmesh::rebase_textures(l, &vis.textures, &vis.data_textures).map_err(err)?;
    }
    let out = Object::new();
    set(&out, "visual", &visual_js(&vis).into());
    set(&out, "visualLod", &visual_js(&lod).into());
    set(&out, "visualFar", &visual_js(&far_vis).into());
    set(
        &out,
        "lodDistance",
        &JsValue::from(flexmesh::lod_distance(flexmesh::LOD_CELL)),
    );
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

/// A vehicle model in the vehicle art spec's glTF binary (`.glb`: an
/// artist's delivery or `gen2`'s `sedan.glb`) as the showcase renderer's visual
/// at rest (`engine_softbody::glb::visual`). `spec`: `{ paint?: [r, g, b] }`
/// (default: the model's own `paint` colour).
/// Returns `{ visual, parts, markers, triangles }`.
#[wasm_bindgen(js_name = glbVisual)]
pub fn glb_visual(bytes: &[u8], spec: JsValue) -> Result<JsValue, JsValue> {
    let v: serde_json::Value = serde_wasm_bindgen::from_value(spec).unwrap_or(serde_json::Value::Null);
    let g = engine_softbody::glb::read(bytes).map_err(err)?;
    let paint = v
        .get("paint")
        .and_then(|p| p.as_array())
        .filter(|a| a.len() == 3)
        .map(|a| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.5) as f32))
        .or_else(|| engine_softbody::glb::paint_of(&g))
        .unwrap_or([0.42, 0.035, 0.03]);
    let vis = engine_softbody::glb::visual(&g, paint);
    let out = Object::new();
    set(&out, "visual", &visual_js(&vis).into());
    set(&out, "parts", &JsValue::from(g.meshes.len() as u32));
    set(&out, "markers", &JsValue::from(g.markers.len() as u32));
    set(&out, "triangles", &JsValue::from((vis.indices.len() / 3) as u32));
    Ok(out.into())
}
