//! Render-ready buffers and glTF 2.0 binary export.
//!
//! Quads split along their shorter diagonal; normals are smooth within 42
//! degrees and split across sharper edges (the Blender exporter's
//! `set_sharp_from_angle(42)`); triangles are grouped by material.

use crate::fx::FxHashMap;


use glam::{DQuat, DVec3};
use serde_json::{json, Value};

use crate::mesh::{key3, Mat, Mesh, MATERIALS};
use crate::uvpack::Parts;

pub const SHARP_DEG: f64 = 42.0;

/// One part as flat render buffers (f32), triangles grouped by material.
#[derive(Clone, Debug, Default)]
pub struct RenderPart {
    pub name: String,
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f32>,
    /// RGBA per vertex, when the part has vertex colours (lamp lenses)
    pub colors: Option<Vec<f32>>,
    /// surface tag per vertex (outer skin / edge wall / inner face)
    pub surf: Vec<crate::mesh::Surf>,
    pub indices: Vec<u32>,
    /// (material, first index, index count)
    pub groups: Vec<(Mat, u32, u32)>,
}

/// A marker: name, position and its local +X axis (glTF coordinates).
#[derive(Clone, Debug)]
pub struct Marker {
    pub name: String,
    pub pos: DVec3,
    pub xaxis: DVec3,
}

/// Triangles of a mesh with per-corner uv and the face's material: quads
/// split along the shorter diagonal.
pub(crate) fn tri_corners(m: &Mesh) -> Vec<([u32; 3], [glam::DVec2; 3], Mat)> {
    let mut out = Vec::with_capacity(m.ntris());
    for (i, f) in m.q.iter().enumerate() {
        let uv = m.quv[i];
        let p = |k: usize| m.v[f[k] as usize];
        let d02 = (p(2) - p(0)).length_squared();
        let d13 = (p(3) - p(1)).length_squared();
        if d02 <= d13 {
            out.push(([f[0], f[1], f[2]], [uv[0], uv[1], uv[2]], m.qm[i]));
            out.push(([f[0], f[2], f[3]], [uv[0], uv[2], uv[3]], m.qm[i]));
        } else {
            out.push(([f[0], f[1], f[3]], [uv[0], uv[1], uv[3]], m.qm[i]));
            out.push(([f[1], f[2], f[3]], [uv[1], uv[2], uv[3]], m.qm[i]));
        }
    }
    for (i, f) in m.t.iter().enumerate() {
        out.push((*f, m.tuv[i], m.tm[i]));
    }
    out
}

/// A mesh as the triangles the export writes (quads split along their
/// shorter diagonal).
pub fn split_triangles(m: &Mesh) -> Vec<[u32; 3]> {
    tri_corners(m).into_iter().map(|t| t.0).collect()
}

/// Flatten a part to render buffers.
pub fn render_part(name: &str, m: &Mesh) -> RenderPart {
    let tris = tri_corners(m);
    // face normals (area weighted) and the faces round each vertex
    let fnrm: Vec<DVec3> = tris
        .iter()
        .map(|(f, _, _)| {
            let (a, b, c) = (m.v[f[0] as usize], m.v[f[1] as usize], m.v[f[2] as usize]);
            (b - a).cross(c - a)
        })
        .collect();
    let funit: Vec<DVec3> = fnrm.iter().map(|n| *n / n.length().max(1e-30)).collect();
    let mut vf: Vec<Vec<u32>> = vec![vec![]; m.v.len()];
    for (k, (f, _, _)) in tris.iter().enumerate() {
        for &i in f {
            vf[i as usize].push(k as u32);
        }
    }
    let cos_sharp = SHARP_DEG.to_radians().cos();
    // group triangles by material, keep a stable order
    let mut order: Vec<usize> = (0..tris.len()).collect();
    order.sort_by_key(|&k| tris[k].2);
    let mut rp = RenderPart { name: name.into(), ..Default::default() };
    let mut colors: Vec<f32> = Vec::new();
    let mut vmap: FxHashMap<(u32, (i64, i64, i64), (i64, i64)), u32> = FxHashMap::default();
    let mut cur: Option<Mat> = None;
    for &k in &order {
        let (f, uv, mat) = &tris[k];
        if cur != Some(*mat) {
            rp.groups.push((*mat, rp.indices.len() as u32, 0));
            cur = Some(*mat);
        }
        for c in 0..3 {
            let vi = f[c];
            let mut n = DVec3::ZERO;
            for &g in &vf[vi as usize] {
                if funit[g as usize].dot(funit[k]) >= cos_sharp {
                    n += fnrm[g as usize];
                }
            }
            let n = if n.length() > 0.0 { n.normalize() } else { funit[k] };
            let key = (vi, key3(n, 1e-4), ((uv[c].x * 1e6).round() as i64, (uv[c].y * 1e6).round() as i64));
            let id = *vmap.entry(key).or_insert_with(|| {
                let p = m.v[vi as usize];
                rp.positions.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
                rp.normals.extend_from_slice(&[n.x as f32, n.y as f32, n.z as f32]);
                rp.uvs.extend_from_slice(&[uv[c].x as f32, uv[c].y as f32]);
                rp.surf.push(m.surf_of(vi as usize));
                if let Some(col) = &m.col {
                    let c = col[vi as usize];
                    colors.extend_from_slice(&[c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32]);
                }
                (rp.positions.len() / 3 - 1) as u32
            });
            rp.indices.push(id);
        }
        rp.groups.last_mut().unwrap().2 += 3;
    }
    if m.col.is_some() {
        rp.colors = Some(colors);
    }
    rp
}

pub fn render_parts(parts: &Parts) -> Vec<RenderPart> {
    parts.0.iter().filter(|(_, m)| m.ntris() > 0).map(|(n, m)| render_part(n, m)).collect()
}

/// PBR factors of the spec's ten materials (paint from the design).
pub fn material_json(paint: [f64; 3], metallic: f64, rough: f64) -> Vec<Value> {
    MATERIALS
        .iter()
        .map(|&name| {
            let (c, m, r, alpha): ([f64; 4], f64, f64, bool) = match name {
                "paint" => ([paint[0], paint[1], paint[2], 1.0], metallic, rough, false),
                "chrome" => ([0.92, 0.92, 0.94, 1.0], 1.0, 0.06, false),
                "trim_black" => ([0.018, 0.018, 0.02, 1.0], 0.0, 0.72, false),
                "rubber" => ([0.016, 0.016, 0.017, 1.0], 0.0, 0.78, false),
                "glass" => ([0.06, 0.075, 0.08, 0.22], 0.0, 0.02, true),
                "lamp_lens" => ([1.0, 1.0, 1.0, 0.45], 0.0, 0.03, true),
                "lamp_emissive" => ([0.9, 0.9, 0.88, 1.0], 0.6, 0.2, false),
                "interior_fabric" => ([0.035, 0.035, 0.037, 1.0], 0.0, 0.9, false),
                "interior_plastic" => ([0.03, 0.03, 0.032, 1.0], 0.0, 0.55, false),
                _ => ([0.30, 0.30, 0.31, 1.0], 1.0, 0.55, false),
            };
            let mut mj = json!({
                "name": name,
                "pbrMetallicRoughness": { "baseColorFactor": c, "metallicFactor": m, "roughnessFactor": r },
                "doubleSided": false
            });
            if alpha {
                mj["alphaMode"] = json!("BLEND");
            }
            mj
        })
        .collect()
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
        let mut v = json!({ "buffer": 0, "byteOffset": off, "byteLength": bytes.len() });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.views.push(v);
        self.views.len() - 1
    }

    fn f32_attr(&mut self, d: &[f32], comps: usize, kind: &str, minmax: bool) -> usize {
        let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
        let view = self.push(&bytes, Some(34962));
        let mut a = json!({ "bufferView": view, "componentType": 5126, "count": d.len() / comps, "type": kind });
        if minmax {
            let mut lo = vec![f32::INFINITY; comps];
            let mut hi = vec![f32::NEG_INFINITY; comps];
            for c in d.chunks(comps) {
                for k in 0..comps {
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

    fn indices(&mut self, d: &[u32]) -> usize {
        let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
        let view = self.push(&bytes, Some(34963));
        self.accessors.push(json!({ "bufferView": view, "componentType": 5125, "count": d.len(), "type": "SCALAR" }));
        self.accessors.len() - 1
    }
}

/// A glTF 2.0 binary: one root node `vehicle`, one child node per part
/// (one primitive per material) and one empty node per marker.
pub fn glb(parts: &[RenderPart], markers: &[Marker], paint: [f64; 3], metallic: f64, rough: f64) -> Vec<u8> {
    let mut bin = Bin { data: vec![], views: vec![], accessors: vec![] };
    let mut meshes = Vec::new();
    let mut nodes = vec![json!({ "name": "vehicle", "children": [] })];
    for p in parts {
        // per-material primitives sharing the part's vertex attributes
        let pos = bin.f32_attr(&p.positions, 3, "VEC3", true);
        let nrm = bin.f32_attr(&p.normals, 3, "VEC3", false);
        let uv = bin.f32_attr(&p.uvs, 2, "VEC2", false);
        let col = p.colors.as_ref().map(|c| bin.f32_attr(c, 4, "VEC4", false));
        // solid parts: 0 outer skin, 1 edge wall, 2 inner face
        let surf = p.surf.iter().any(|s| *s != crate::mesh::Surf::Outer).then(|| {
            let d: Vec<f32> = p.surf.iter().map(|s| *s as u8 as f32).collect();
            bin.f32_attr(&d, 1, "SCALAR", false)
        });
        let mut prims = Vec::new();
        for &(mat, start, count) in &p.groups {
            let idx = bin.indices(&p.indices[start as usize..(start + count) as usize]);
            let mut attrs = json!({ "POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": uv });
            if let Some(c) = col {
                attrs["COLOR_0"] = json!(c);
            }
            if let Some(s) = surf {
                attrs["_SURFACE"] = json!(s);
            }
            prims.push(json!({ "attributes": attrs, "indices": idx, "material": mat as usize }));
        }
        meshes.push(json!({ "name": p.name, "primitives": prims }));
        nodes.push(json!({ "name": p.name, "mesh": meshes.len() - 1 }));
    }
    for mk in markers {
        // rotation about +Y taking local +X onto the marker's axis
        let ang = (-mk.xaxis.z).atan2(mk.xaxis.x);
        let q = DQuat::from_rotation_y(ang);
        nodes.push(json!({
            "name": mk.name,
            "translation": [mk.pos.x, mk.pos.y, mk.pos.z],
            "rotation": [q.x, q.y, q.z, q.w]
        }));
    }
    let children: Vec<usize> = (1..nodes.len()).collect();
    nodes[0]["children"] = json!(children);
    while bin.data.len() % 4 != 0 {
        bin.data.push(0);
    }
    let doc = json!({
        "asset": { "version": "2.0", "generator": "engine-vehiclegen" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": material_json(paint, metallic, rough),
        "accessors": bin.accessors,
        "bufferViews": bin.views,
        "buffers": [{ "byteLength": bin.data.len() }]
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
