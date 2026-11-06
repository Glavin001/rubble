//! glTF 2.0 binary (`.glb`) vehicles in the vehicle art spec's layout: one
//! mesh node per part under a root named `vehicle`, markers as mesh-less
//! nodes, the spec's PBR materials. [`write`] and [`read`] are a minimal
//! writer and reader (float positions, normals, UV0, COLOR_0; u16 or u32
//! indices; node TRS or matrices), and [`visual`] turns a model into the
//! showcase renderer's visual at rest (every vertex rigid on a fixed frame),
//! so an artist's file and Rubble's generated car load the same way.
//!
//! Frame: glTF's (+Y up, the car's front toward +Z), which is the renderer's.

use crate::authoring::skin::{mat, ShowcaseVisual};
use glam::{Mat4, Quat, Vec3};
use serde_json::{json, Value};

#[derive(Clone, Debug, Default)]
pub struct Prim {
    pub material: usize,
    pub pos: Vec<[f32; 3]>,
    pub nrm: Vec<[f32; 3]>,
    pub uv: Vec<[f32; 2]>,
    /// Per vertex colour (empty: white).
    pub col: Vec<[f32; 3]>,
    pub idx: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub name: String,
    pub prims: Vec<Prim>,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub base: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub blend: bool,
    pub emissive: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Marker {
    pub name: String,
    pub pos: [f32; 3],
    /// Rotation quaternion (x, y, z, w).
    pub rot: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct Glb {
    pub root: String,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub markers: Vec<Marker>,
}

struct Bin {
    data: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Bin {
    fn push(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.data.len() % 4 != 0 {
            self.data.push(0);
        }
        let off = self.data.len();
        self.data.extend_from_slice(bytes);
        let mut v = json!({"buffer": 0, "byteOffset": off, "byteLength": bytes.len()});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.views.push(v);
        self.views.len() - 1
    }
    fn floats(&mut self, data: &[f32], width: usize, kind: &str, minmax: bool) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = self.push(&bytes, Some(34962));
        let count = data.len() / width;
        let mut a =
            json!({"bufferView": view, "componentType": 5126, "count": count, "type": kind});
        if minmax && count > 0 {
            let mut lo = vec![f32::MAX; width];
            let mut hi = vec![f32::MIN; width];
            for c in data.chunks(width) {
                for k in 0..width {
                    lo[k] = lo[k].min(c[k]);
                    hi[k] = hi[k].max(c[k]);
                }
            }
            a["min"] = json!(lo);
            a["max"] = json!(hi);
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }
    fn indices(&mut self, idx: &[u32]) -> usize {
        let bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
        let view = self.push(&bytes, Some(34963));
        self.accessors.push(json!({"bufferView": view, "componentType": 5125, "count": idx.len(), "type": "SCALAR"}));
        self.accessors.len() - 1
    }
}

/// Write a `.glb`: the root node, one node per mesh, then the markers.
pub fn write(g: &Glb) -> Vec<u8> {
    let mut bin = Bin {
        data: Vec::new(),
        views: Vec::new(),
        accessors: Vec::new(),
    };
    let mut meshes = Vec::new();
    let mut nodes = vec![json!({"name": g.root})];
    let mut children = Vec::new();
    for m in &g.meshes {
        let mut prims = Vec::new();
        for p in &m.prims {
            let pos = bin.floats(&p.pos.concat(), 3, "VEC3", true);
            let nrm = bin.floats(&p.nrm.concat(), 3, "VEC3", false);
            let uv = bin.floats(&p.uv.concat(), 2, "VEC2", false);
            let mut attrs = json!({"POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": uv});
            if !p.col.is_empty() {
                attrs["COLOR_0"] = json!(bin.floats(&p.col.concat(), 3, "VEC3", false));
            }
            let idx = bin.indices(&p.idx);
            prims.push(
                json!({"attributes": attrs, "indices": idx, "material": p.material, "mode": 4}),
            );
        }
        meshes.push(json!({"name": m.name, "primitives": prims}));
        nodes.push(json!({"name": m.name, "mesh": meshes.len() - 1}));
        children.push(nodes.len() - 1);
    }
    for k in &g.markers {
        nodes.push(json!({"name": k.name, "translation": k.pos, "rotation": k.rot}));
        children.push(nodes.len() - 1);
    }
    nodes[0]["children"] = json!(children);
    let materials: Vec<Value> = g
        .materials
        .iter()
        .map(|m| {
            let mut v = json!({
                "name": m.name,
                "pbrMetallicRoughness": {"baseColorFactor": m.base, "metallicFactor": m.metallic, "roughnessFactor": m.roughness},
                "doubleSided": m.blend,
            });
            if m.blend {
                v["alphaMode"] = json!("BLEND");
            }
            if m.emissive != [0.0; 3] {
                v["emissiveFactor"] = json!(m.emissive);
            }
            v
        })
        .collect();
    while bin.data.len() % 4 != 0 {
        bin.data.push(0);
    }
    let doc = json!({
        "asset": {"version": "2.0", "generator": "Rubble gen2 (engine_softbody::authoring::gen2::model)"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": bin.accessors,
        "bufferViews": bin.views,
        "buffers": [{"byteLength": bin.data.len()}],
    });
    let mut js = serde_json::to_vec(&doc).unwrap();
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin.data.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(js.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&js);
    out.extend_from_slice(&(bin.data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin.data);
    out
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Read a `.glb`: every mesh node in world space (its transforms applied),
/// mesh-less nodes as markers.
pub fn read(bytes: &[u8]) -> Result<Glb, String> {
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return Err("glb: not a glTF binary".into());
    }
    let jl = u32_at(bytes, 12) as usize;
    if &bytes[16..20] != b"JSON" {
        return Err("glb: first chunk is not JSON".into());
    }
    let doc: Value =
        serde_json::from_slice(&bytes[20..20 + jl]).map_err(|e| format!("glb json: {e}"))?;
    let bo = 20 + jl;
    let bin: &[u8] = if bytes.len() >= bo + 8 {
        let bl = u32_at(bytes, bo) as usize;
        &bytes[bo + 8..bo + 8 + bl]
    } else {
        &[]
    };
    let arr = |k: &str| {
        doc.get(k)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let (accessors, views, nodes, meshes, mats) = (
        arr("accessors"),
        arr("bufferViews"),
        arr("nodes"),
        arr("meshes"),
        arr("materials"),
    );
    let read_acc = |i: usize| -> Result<(Vec<f32>, usize), String> {
        let a = accessors.get(i).ok_or("glb: accessor")?;
        let width = match a["type"].as_str().unwrap_or("") {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            t => return Err(format!("glb: accessor type {t}")),
        };
        let count = a["count"].as_u64().unwrap_or(0) as usize;
        let ct = a["componentType"].as_u64().unwrap_or(0);
        let v = &views[a["bufferView"]
            .as_u64()
            .ok_or("glb: sparse accessors are not read")? as usize];
        let off = v["byteOffset"].as_u64().unwrap_or(0) as usize
            + a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let csize = match ct {
            5126 | 5125 => 4,
            5123 | 5122 => 2,
            5121 | 5120 => 1,
            _ => return Err(format!("glb: component type {ct}")),
        };
        let stride = v["byteStride"]
            .as_u64()
            .map_or(width * csize, |s| s as usize);
        let norm = a["normalized"].as_bool().unwrap_or(false);
        let mut out = Vec::with_capacity(count * width);
        for e in 0..count {
            for c in 0..width {
                let o = off + e * stride + c * csize;
                let x = match ct {
                    5126 => f32::from_le_bytes([bin[o], bin[o + 1], bin[o + 2], bin[o + 3]]),
                    5125 => u32_at(bin, o) as f32,
                    5123 => {
                        let u = u16::from_le_bytes([bin[o], bin[o + 1]]) as f32;
                        if norm {
                            u / 65535.0
                        } else {
                            u
                        }
                    }
                    5121 => {
                        let u = bin[o] as f32;
                        if norm {
                            u / 255.0
                        } else {
                            u
                        }
                    }
                    _ => 0.0,
                };
                out.push(x);
            }
        }
        Ok((out, width))
    };
    let read_idx = |i: usize| -> Result<Vec<u32>, String> {
        let a = &accessors[i];
        let count = a["count"].as_u64().unwrap_or(0) as usize;
        let ct = a["componentType"].as_u64().unwrap_or(0);
        let v = &views[a["bufferView"].as_u64().unwrap_or(0) as usize];
        let off = v["byteOffset"].as_u64().unwrap_or(0) as usize
            + a["byteOffset"].as_u64().unwrap_or(0) as usize;
        Ok((0..count)
            .map(|e| match ct {
                5125 => u32_at(bin, off + 4 * e),
                5123 => u16::from_le_bytes([bin[off + 2 * e], bin[off + 2 * e + 1]]) as u32,
                _ => bin[off + e] as u32,
            })
            .collect())
    };
    let materials: Vec<Material> = mats
        .iter()
        .map(|m| {
            let pbr = &m["pbrMetallicRoughness"];
            let f4 = |v: &Value, d: [f32; 4]| -> [f32; 4] {
                v.as_array().filter(|a| a.len() == 4).map_or(d, |a| {
                    [0, 1, 2, 3].map(|k| a[k].as_f64().unwrap_or(1.0) as f32)
                })
            };
            Material {
                name: m["name"].as_str().unwrap_or("").to_string(),
                base: f4(&pbr["baseColorFactor"], [1.0; 4]),
                metallic: pbr["metallicFactor"].as_f64().unwrap_or(1.0) as f32,
                roughness: pbr["roughnessFactor"].as_f64().unwrap_or(1.0) as f32,
                blend: m["alphaMode"].as_str() == Some("BLEND"),
                emissive: m["emissiveFactor"]
                    .as_array()
                    .filter(|a| a.len() == 3)
                    .map_or([0.0; 3], |a| {
                        [0, 1, 2].map(|k| a[k].as_f64().unwrap_or(0.0) as f32)
                    }),
            }
        })
        .collect();
    // Node world matrices from the scene roots down.
    let local = |n: &Value| -> Mat4 {
        if let Some(m) = n["matrix"].as_array().filter(|a| a.len() == 16) {
            let v: Vec<f32> = m.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            return Mat4::from_cols_slice(&v);
        }
        let f = |k: &str, d: Vec<f32>| -> Vec<f32> {
            n[k].as_array().map_or(d, |a| {
                a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()
            })
        };
        let t = f("translation", vec![0.0; 3]);
        let r = f("rotation", vec![0.0, 0.0, 0.0, 1.0]);
        let s = f("scale", vec![1.0; 3]);
        Mat4::from_scale_rotation_translation(
            Vec3::new(s[0], s[1], s[2]),
            Quat::from_xyzw(r[0], r[1], r[2], r[3]),
            Vec3::new(t[0], t[1], t[2]),
        )
    };
    let mut world = vec![None::<Mat4>; nodes.len()];
    let roots: Vec<usize> = doc["scenes"][doc["scene"].as_u64().unwrap_or(0) as usize]["nodes"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_else(|| (0..nodes.len()).collect());
    let mut stack: Vec<(usize, Mat4)> = roots.iter().map(|&r| (r, Mat4::IDENTITY)).collect();
    let mut root_name = String::new();
    while let Some((i, parent)) = stack.pop() {
        let w = parent * local(&nodes[i]);
        world[i] = Some(w);
        if root_name.is_empty() {
            root_name = nodes[i]["name"].as_str().unwrap_or("").to_string();
        }
        for c in nodes[i]["children"].as_array().into_iter().flatten() {
            if let Some(c) = c.as_u64() {
                stack.push((c as usize, w));
            }
        }
    }
    let mut out = Glb {
        root: root_name,
        materials,
        ..Default::default()
    };
    for (i, n) in nodes.iter().enumerate() {
        let Some(w) = world[i] else { continue };
        let name = n["name"].as_str().unwrap_or("").to_string();
        let Some(mi) = n["mesh"].as_u64() else {
            if i != roots.first().copied().unwrap_or(usize::MAX) {
                let (_, r, t) = w.to_scale_rotation_translation();
                out.markers.push(Marker {
                    name,
                    pos: t.to_array(),
                    rot: r.to_array(),
                });
            }
            continue;
        };
        let nm = w.inverse().transpose();
        let mut mesh = Mesh {
            name,
            prims: Vec::new(),
        };
        for p in meshes[mi as usize]["primitives"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if p["mode"].as_u64().unwrap_or(4) != 4 {
                continue;
            }
            let at = &p["attributes"];
            let get = |k: &str| at[k].as_u64().map(|i| read_acc(i as usize));
            let (pos, _) = get("POSITION").ok_or("glb: a primitive without POSITION")??;
            let pos: Vec<[f32; 3]> = pos
                .chunks(3)
                .map(|c| w.transform_point3(Vec3::new(c[0], c[1], c[2])).to_array())
                .collect();
            let nrm: Vec<[f32; 3]> = match get("NORMAL") {
                Some(r) => {
                    r?.0.chunks(3)
                        .map(|c| {
                            nm.transform_vector3(Vec3::new(c[0], c[1], c[2]))
                                .normalize_or(Vec3::Y)
                                .to_array()
                        })
                        .collect()
                }
                None => vec![[0.0, 1.0, 0.0]; pos.len()],
            };
            let uv: Vec<[f32; 2]> = match get("TEXCOORD_0") {
                Some(r) => r?.0.chunks(2).map(|c| [c[0], c[1]]).collect(),
                None => vec![[0.0; 2]; pos.len()],
            };
            let col: Vec<[f32; 3]> = match get("COLOR_0") {
                Some(r) => {
                    let (v, wd) = r?;
                    v.chunks(wd).map(|c| [c[0], c[1], c[2]]).collect()
                }
                None => Vec::new(),
            };
            let idx = match p["indices"].as_u64() {
                Some(i) => read_idx(i as usize)?,
                None => (0..pos.len() as u32).collect(),
            };
            mesh.prims.push(Prim {
                material: p["material"].as_u64().map_or(usize::MAX, |m| m as usize),
                pos,
                nrm,
                uv,
                col,
                idx,
            });
        }
        out.meshes.push(mesh);
    }
    Ok(out)
}

/// The renderer's material class for a spec material on a part.
pub fn class_of(material: &str, part: &str) -> u8 {
    let p = part.to_ascii_lowercase();
    let tail = p.starts_with("taillight");
    match material {
        "paint" => mat::PAINT,
        "chrome" => mat::CHROME,
        "trim_black" => {
            if p.starts_with("underbody") || p.starts_with("wheel_well") {
                mat::UNDERBODY
            } else if p.starts_with("grille") {
                mat::GRILLE
            } else if tail && p.ends_with("housing") {
                mat::TAIL_BACK
            } else {
                mat::TRIM
            }
        }
        "rubber" => mat::TYRE,
        "glass" => {
            if p.contains("windshield") {
                mat::SCREEN
            } else if p.contains("rear")
                || p.contains("quarter")
                || p.ends_with("_rl")
                || p.ends_with("_rr")
            {
                mat::PRIVACY
            } else {
                mat::GLASS
            }
        }
        "lamp_lens" => {
            if tail {
                mat::TAIL_LENS
            } else {
                mat::LENS
            }
        }
        "lamp_emissive" => {
            if tail {
                mat::TAIL_LED
            } else {
                mat::DRL
            }
        }
        "interior_fabric" => {
            if p.starts_with("headliner") {
                mat::HEADLINER
            } else if p.starts_with("console") {
                mat::CARPET
            } else {
                mat::SEAT
            }
        }
        "interior_plastic" => {
            if p.starts_with("door_trim") {
                mat::DOORCARD
            } else {
                mat::DASH
            }
        }
        "metal_bare" => {
            if p.starts_with("rim") {
                mat::RIM
            } else if p.starts_with("brake") {
                mat::BRAKE
            } else if p.starts_with("firewall") {
                mat::INNER
            } else if p.starts_with("trunk_lid") || p.starts_with("bumper") {
                // Bare metal on a closure or bumper: the licence plate.
                mat::PLATE
            } else {
                mat::ENGINE
            }
        }
        other => crate::flexmesh::material_class(other),
    }
}

/// A visual of the model at rest: every vertex rigid on one fixed frame of
/// four nodes (no simulation), shaded by its material's class and colour
/// (paint takes `paint`).
pub fn visual(g: &Glb, paint: [f32; 3]) -> ShowcaseVisual {
    let nodes = [
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(1.0, 0.5, 0.0),
        Vec3::new(0.0, 0.5, 1.0),
        Vec3::new(-1.0, 0.9, -1.0),
    ];
    let (t, b, n) = (Vec3::X, Vec3::Z, -Vec3::Y);
    let mut v = ShowcaseVisual {
        node_count: nodes.len() as u32,
        node_rest: nodes.iter().map(|p| p.to_array()).collect(),
        ..Default::default()
    };
    for m in &g.meshes {
        let first = v.positions.len() as u32;
        for p in &m.prims {
            let base = v.positions.len() as u32;
            let mi = g.materials.get(p.material);
            let mname = mi.map_or("", |x| x.name.as_str());
            let class = class_of(mname, &m.name);
            let bc = mi.map_or([1.0; 4], |x| x.base);
            for k in 0..p.pos.len() {
                let q = Vec3::from_array(p.pos[k]) - nodes[0];
                let nn = Vec3::from_array(p.nrm[k]);
                v.positions.push(p.pos[k]);
                v.normals.push(p.nrm[k]);
                v.uvs.push(p.uv[k]);
                v.bind_nodes.push([0, 1, 2]);
                v.bind_coords.push([q.dot(t), q.dot(b), q.dot(n)]);
                v.bind_normal.push([nn.dot(t), nn.dot(b), nn.dot(n)]);
                v.bind_rest.push([1.0, 1.0]);
                v.ao.push(1.0);
                v.smooth.push(0);
                v.rigid.push(1);
                v.material.push(class);
                let c = p.col.get(k).copied().unwrap_or([1.0; 3]);
                v.colors.push(if class == mat::PAINT {
                    paint
                } else {
                    [bc[0] * c[0], bc[1] * c[1], bc[2] * c[2]]
                });
                v.tex.push(u16::MAX);
            }
            v.indices.extend(p.idx.iter().map(|i| i + base));
        }
        let count = v.positions.len() as u32 - first;
        v.pieces.push((m.name.clone(), first, count));
    }
    let nv = v.positions.len();
    v.unit = vec![u32::MAX; nv];
    v.shard = vec![0; nv];
    v.wheel = vec![u32::MAX; nv];
    v.part = vec![u16::MAX; nv];
    let nn = nodes.len();
    v.node_tri_start = vec![0; nn + 1];
    v.node_rest_normal = vec![[0.0, 1.0, 0.0]; nn];
    v.node_beam_start = vec![0; nn + 1];
    v.node_material = vec![crate::authoring::skin::node_mat::METAL; nn];
    v
}
