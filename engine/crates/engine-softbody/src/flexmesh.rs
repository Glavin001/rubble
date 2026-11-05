//! The visual layer of an imported BeamNG JBeam vehicle: its flexbodies
//! (Collada `.dae` meshes bound to node groups) and props (rigid meshes on
//! three nodes), as a [`ShowcaseVisual`], so the showcase renderer
//! (`showcase-car-render.mjs`) skins and shades it as it does the authored car.
//!
//! Clean room: written from BeamNG's public documentation of the `flexbodies`
//! and `props` sections (documentation.beamng.com, "Flexbodies", "Props")
//! and the Collada 1.4 specification; no BeamNG code read.
//!
//! * **Flexbodies.** Each row names a mesh (a Collada scene node, or a
//!   geometry, by name) and the node groups its vertices bind to. Every vertex
//!   binds to three of those groups' nodes: the nearest node and the pair of
//!   others that makes the most perpendicular frame ([`crate::flexbody::bind`]),
//!   so it rides the part it belongs to and a detached part carries its mesh.
//!   A row's `pos`, `rot` (degrees) and `scale` options move the mesh.
//! * **Props.** Each row puts a rigid mesh on its `idRef`, `idX`, `idY`
//!   nodes: it rides their frame without stretching.
//! * **Materials.** BeamNG's materials are textures; Rubble's renderer shades
//!   by material class. A mesh's Collada material name picks the class
//!   ([`material_class`]: glass, lamp lens, chrome, tyre, rim, trim, interior,
//!   else paint).
//!
//! Coordinates: the meshes are in the JBeam frame (+X left, +Y rear, +Z up,
//! Collada `Z_UP`); they come out in Rubble's (+Y up, +Z forward), as the
//! nodes do ([`crate::jbeam::to_rubble`]).

use crate::authoring::skin::{mat, ShowcaseVisual};
use crate::desc::SoftVehicleDesc;
use glam::{Mat4, Quat, Vec3};
use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;
use serde_json::{Map, Value};
use std::collections::HashMap;

/// One mesh of a Collada file: a scene node's geometry with its node's
/// transform applied (or a geometry no node instances, untransformed).
#[derive(Clone, Debug, Default)]
pub struct DaeMesh {
    /// The scene node's name (BeamNG's mesh name), else the geometry's.
    pub name: String,
    pub geometry: String,
    pub pos: Vec<Vec3>,
    pub nrm: Vec<Vec3>,
    pub uv: Vec<[f32; 2]>,
    pub tris: Vec<[u32; 3]>,
    /// Per triangle its material (an index into `materials`).
    pub tri_mat: Vec<u16>,
    pub materials: Vec<String>,
}

#[derive(Default)]
struct Prim {
    material: String,
    /// (semantic, source id, offset, set)
    inputs: Vec<(String, String, usize, u32)>,
    vcount: Vec<u32>,
    p: Vec<u32>,
    triangles: bool,
}

#[derive(Default)]
struct Geom {
    name: String,
    prims: Vec<Prim>,
    /// `<vertices>` id → its POSITION (and NORMAL, TEXCOORD) sources.
    vertices: HashMap<String, Vec<(String, String)>>,
}

struct SceneNode {
    name: String,
    world: Mat4,
    /// (geometry id, material symbol → material id)
    geoms: Vec<(String, HashMap<String, String>)>,
}

fn attr(e: &BytesStart, key: &[u8]) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| a.key.as_ref() == key)
        .map(|a| String::from_utf8_lossy(&a.value).into_owned())
}

fn floats(s: &str) -> Vec<f32> {
    s.split_ascii_whitespace()
        .map(|x| x.parse::<f32>().unwrap_or(0.0))
        .collect()
}

fn uints(s: &str) -> Vec<u32> {
    s.split_ascii_whitespace()
        .map(|x| x.parse::<u32>().unwrap_or(0))
        .collect()
}

/// Read a Collada 1.4 document's meshes (triangles and polylists; polygons
/// fanned), transformed by their scene nodes.
pub fn read_collada(src: &str) -> Result<Vec<DaeMesh>, String> {
    let mut r = Reader::from_str(src);
    r.config_mut().trim_text(true);
    let mut stack: Vec<String> = Vec::new();
    let mut sources: HashMap<String, (Vec<f32>, usize)> = HashMap::new();
    let mut geoms: HashMap<String, Geom> = HashMap::new();
    let mut geom_order: Vec<String> = Vec::new();
    let mut mat_names: HashMap<String, String> = HashMap::new();
    let mut nodes: Vec<SceneNode> = Vec::new();
    let mut node_stack: Vec<usize> = Vec::new();
    let mut z_up = true;
    // Parse state.
    let mut cur_geom: Option<String> = None;
    let mut cur_source: Option<String> = None;
    let mut cur_vertices: Option<String> = None;
    let mut cur_prim: Option<Prim> = None;
    let mut cur_float: Vec<f32> = Vec::new();
    let mut text = String::new();
    let mut inst_geom: Option<(String, HashMap<String, String>)> = None;
    loop {
        let ev = r
            .read_event()
            .map_err(|e| format!("dae: at {}: {e}", r.error_position()))?;
        let (start, empty) = match &ev {
            Event::Start(_) => (true, false),
            Event::Empty(_) => (true, true),
            _ => (false, false),
        };
        match ev {
            Event::Eof => break,
            Event::Start(ref e) | Event::Empty(ref e) if start => {
                let name = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
                match name.as_str() {
                    "geometry" => {
                        let id = attr(e, b"id").unwrap_or_default();
                        geoms.insert(
                            id.clone(),
                            Geom {
                                name: attr(e, b"name").unwrap_or_else(|| id.clone()),
                                ..Default::default()
                            },
                        );
                        geom_order.push(id.clone());
                        cur_geom = Some(id);
                    }
                    "source" if cur_geom.is_some() => cur_source = attr(e, b"id"),
                    "accessor" => {
                        if let Some(s) = &cur_source {
                            let stride = attr(e, b"stride").and_then(|x| x.parse().ok()).unwrap_or(1);
                            sources
                                .entry(s.clone())
                                .or_insert_with(|| (Vec::new(), stride))
                                .1 = stride;
                        }
                    }
                    "vertices" => cur_vertices = attr(e, b"id"),
                    "triangles" | "polylist" | "polygons" => {
                        cur_prim = Some(Prim {
                            material: attr(e, b"material").unwrap_or_default(),
                            triangles: name == "triangles",
                            ..Default::default()
                        });
                    }
                    "input" => {
                        let sem = attr(e, b"semantic").unwrap_or_default();
                        let src = attr(e, b"source").unwrap_or_default().trim_start_matches('#').to_string();
                        if let Some(p) = cur_prim.as_mut() {
                            let off = attr(e, b"offset").and_then(|x| x.parse().ok()).unwrap_or(0);
                            let set = attr(e, b"set").and_then(|x| x.parse().ok()).unwrap_or(0);
                            p.inputs.push((sem, src, off, set));
                        } else if let (Some(v), Some(g)) = (&cur_vertices, &cur_geom) {
                            geoms
                                .get_mut(g)
                                .unwrap()
                                .vertices
                                .entry(v.clone())
                                .or_default()
                                .push((sem, src));
                        }
                    }
                    "material" if stack.last().is_some_and(|s| s == "library_materials") => {
                        if let Some(id) = attr(e, b"id") {
                            let nm = attr(e, b"name").unwrap_or_else(|| id.clone());
                            mat_names.insert(id, nm);
                        }
                    }
                    "node" => {
                        // A node's transforms come before its children, so a
                        // child starts from its parent's world as it stands.
                        let parent = node_stack.last().map_or(Mat4::IDENTITY, |&n| nodes[n].world);
                        nodes.push(SceneNode {
                            name: attr(e, b"name").or_else(|| attr(e, b"id")).unwrap_or_default(),
                            world: parent,
                            geoms: Vec::new(),
                        });
                        if !empty {
                            node_stack.push(nodes.len() - 1);
                        }
                    }
                    "instance_geometry" => {
                        let url = attr(e, b"url").unwrap_or_default().trim_start_matches('#').to_string();
                        inst_geom = Some((url, HashMap::new()));
                        if empty {
                            if let (Some(g), Some(&ni)) = (inst_geom.take(), node_stack.last()) {
                                nodes[ni].geoms.push(g);
                            }
                        }
                    }
                    "instance_material" => {
                        if let Some((_, m)) = inst_geom.as_mut() {
                            if let (Some(sym), Some(t)) = (attr(e, b"symbol"), attr(e, b"target")) {
                                m.insert(sym, t.trim_start_matches('#').to_string());
                            }
                        }
                    }
                    _ => {}
                }
                if !empty {
                    stack.push(name);
                    text.clear();
                }
            }
            Event::Text(t) => {
                let s = t.decode().map_err(|e| format!("dae: {e}"))?;
                text.push_str(&s);
                text.push(' ');
            }
            Event::End(e) => {
                let name = String::from_utf8_lossy(e.local_name().as_ref()).into_owned();
                stack.pop();
                match name.as_str() {
                    "up_axis" => z_up = !text.trim().eq_ignore_ascii_case("Y_UP"),
                    "float_array" => cur_float = floats(&text),
                    "source" => {
                        if let Some(s) = cur_source.take() {
                            let stride = sources.get(&s).map_or(1, |x| x.1);
                            sources.insert(s, (std::mem::take(&mut cur_float), stride));
                        }
                    }
                    "vertices" => cur_vertices = None,
                    "vcount" => {
                        if let Some(p) = cur_prim.as_mut() {
                            p.vcount = uints(&text);
                        }
                    }
                    "p" => {
                        if let Some(p) = cur_prim.as_mut() {
                            let v = uints(&text);
                            if stack.last().is_some_and(|s| s == "polygons") {
                                // <polygons>: one <p> per polygon.
                                let stride = p.inputs.iter().map(|i| i.2).max().unwrap_or(0) + 1;
                                p.vcount.push((v.len() / stride) as u32);
                            }
                            p.p.extend(v);
                        }
                    }
                    "triangles" | "polylist" | "polygons" => {
                        if let (Some(p), Some(g)) = (cur_prim.take(), &cur_geom) {
                            geoms.get_mut(g).unwrap().prims.push(p);
                        }
                    }
                    "geometry" => cur_geom = None,
                    "matrix" => {
                        if let Some(&ni) = node_stack.last() {
                            let m = floats(&text);
                            if m.len() == 16 {
                                // Collada matrices are row-major.
                                let local = Mat4::from_cols_array(&[
                                    m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6],
                                    m[10], m[14], m[3], m[7], m[11], m[15],
                                ]);
                                nodes[ni].world *= local;
                            }
                        }
                    }
                    "translate" => {
                        if let Some(&ni) = node_stack.last() {
                            let t = floats(&text);
                            if t.len() == 3 {
                                nodes[ni].world *= Mat4::from_translation(Vec3::new(t[0], t[1], t[2]));
                            }
                        }
                    }
                    "rotate" => {
                        if let Some(&ni) = node_stack.last() {
                            let t = floats(&text);
                            if t.len() == 4 {
                                let axis = Vec3::new(t[0], t[1], t[2]).normalize_or(Vec3::Z);
                                nodes[ni].world *= Mat4::from_quat(Quat::from_axis_angle(axis, t[3].to_radians()));
                            }
                        }
                    }
                    "scale" => {
                        if let Some(&ni) = node_stack.last() {
                            let t = floats(&text);
                            if t.len() == 3 {
                                nodes[ni].world *= Mat4::from_scale(Vec3::new(t[0], t[1], t[2]));
                            }
                        }
                    }
                    "instance_geometry" => {
                        if let (Some(g), Some(&ni)) = (inst_geom.take(), node_stack.last()) {
                            nodes[ni].geoms.push(g);
                        }
                    }
                    "node" => {
                        node_stack.pop();
                    }
                    _ => {}
                }
                text.clear();
            }
            _ => {}
        }
    }
    let up = |p: Vec3| if z_up { p } else { Vec3::new(p.x, -p.z, p.y) };
    let build = |gid: &str, name: &str, world: Mat4, bind: &HashMap<String, String>| -> Option<DaeMesh> {
        let g = geoms.get(gid)?;
        let mut out = DaeMesh {
            name: name.to_string(),
            geometry: g.name.clone(),
            ..Default::default()
        };
        let nm = Mat4::from_mat3(glam::Mat3::from_mat4(world).inverse().transpose());
        let mut cache: HashMap<Vec<u32>, u32> = HashMap::new();
        for p in &g.prims {
            let mname = bind
                .get(&p.material)
                .map(|id| mat_names.get(id).cloned().unwrap_or_else(|| id.clone()))
                .unwrap_or_else(|| p.material.clone());
            let mi = out.materials.iter().position(|m| *m == mname).unwrap_or_else(|| {
                out.materials.push(mname.clone());
                out.materials.len() - 1
            }) as u16;
            let stride = p.inputs.iter().map(|i| i.2).max().unwrap_or(0) + 1;
            // Resolve each input to (kind, source): VERTEX expands to its
            // <vertices>' sources at the same offset.
            let mut ins: Vec<(&str, &str, usize)> = Vec::new();
            for (sem, src, off, set) in &p.inputs {
                if sem == "VERTEX" {
                    for (s2, src2) in g.vertices.get(src).into_iter().flatten() {
                        ins.push((s2.as_str(), src2.as_str(), *off));
                    }
                } else if sem != "TEXCOORD" || *set == 0 {
                    ins.push((sem.as_str(), src.as_str(), *off));
                }
            }
            let corners = p.p.len() / stride;
            let mut vid = Vec::with_capacity(corners);
            for c in 0..corners {
                let key = &p.p[c * stride..c * stride + stride];
                let mut pos = Vec3::ZERO;
                let mut nrm = Vec3::Z;
                let mut uv = [0.0; 2];
                for &(sem, src, off) in &ins {
                    let Some((f, st)) = sources.get(src) else { continue };
                    let k = key[off] as usize * st;
                    let g3 = |j: usize| f.get(k + j).copied().unwrap_or(0.0);
                    match sem {
                        "POSITION" => pos = Vec3::new(g3(0), g3(1), g3(2)),
                        "NORMAL" => nrm = Vec3::new(g3(0), g3(1), g3(2)),
                        "TEXCOORD" => uv = [g3(0), g3(1)],
                        _ => {}
                    }
                }
                // Corners equal in position, normal and texture coordinates
                // are one vertex (exporters often write every corner on its
                // own; welding them is exact).
                let wk: Vec<u32> = [pos.x, pos.y, pos.z, nrm.x, nrm.y, nrm.z, uv[0], uv[1]].iter().map(|x| x.to_bits()).collect();
                let id = *cache.entry(wk).or_insert_with(|| {
                    out.pos.push(up(world.transform_point3(pos)));
                    out.nrm.push(up(nm.transform_vector3(nrm)).normalize_or(Vec3::Z));
                    out.uv.push(uv);
                    (out.pos.len() - 1) as u32
                });
                vid.push(id);
            }
            if p.triangles {
                for t in vid.chunks(3) {
                    if t.len() == 3 {
                        out.tris.push([t[0], t[1], t[2]]);
                        out.tri_mat.push(mi);
                    }
                }
            } else {
                // Polylist: fan each polygon (vcount corners).
                let counts: Vec<u32> = if p.vcount.iter().any(|&c| c > 0) {
                    p.vcount.clone()
                } else {
                    vec![3; vid.len() / 3]
                };
                let mut at = 0usize;
                for &n in &counts {
                    let n = n as usize;
                    for k in 1..n.saturating_sub(1) {
                        if at + k + 1 < vid.len() {
                            out.tris.push([vid[at], vid[at + k], vid[at + k + 1]]);
                            out.tri_mat.push(mi);
                        }
                    }
                    at += n;
                }
            }
        }
        Some(out)
    };
    let mut out = Vec::new();
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    for n in &nodes {
        for (gid, bind) in &n.geoms {
            used.insert(gid.clone());
            if let Some(m) = build(gid, &n.name, n.world, bind) {
                out.push(m);
            }
        }
    }
    for gid in &geom_order {
        if !used.contains(gid) {
            let name = geoms[gid].name.clone();
            if let Some(m) = build(gid, &name, Mat4::IDENTITY, &HashMap::new()) {
                out.push(m);
            }
        }
    }
    Ok(out)
}

/// A coarser copy of `m` for the distant level of detail: vertices clustered
/// on a `cell` (m) grid, per material and normal direction (so creases and
/// material borders stay), each cluster at its vertices' mean, triangles that
/// collapse dropped. At a distance where `cell` is about a pixel this is the
/// same image; the near level is always the full mesh.
pub fn cluster(m: &DaeMesh, cell: f32) -> DaeMesh {
    // Per vertex its material (its first triangle's).
    let mut vmat = vec![u16::MAX; m.pos.len()];
    for (t, tri) in m.tris.iter().enumerate() {
        for &v in tri {
            if vmat[v as usize] == u16::MAX {
                vmat[v as usize] = m.tri_mat[t];
            }
        }
    }
    let dir = |n: Vec3| -> u8 {
        // One of 26 directions (each axis -, 0, +).
        let q = |x: f32| if x > 0.38 { 2u8 } else if x < -0.38 { 0 } else { 1 };
        q(n.x) * 9 + q(n.y) * 3 + q(n.z)
    };
    let mut ids: HashMap<(i32, i32, i32, u16, u8), u32> = HashMap::new();
    let mut acc: Vec<(Vec3, Vec3, [f32; 2], u32)> = Vec::new();
    let mut map = vec![0u32; m.pos.len()];
    for (i, p) in m.pos.iter().enumerate() {
        let k = (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
            vmat[i],
            dir(m.nrm[i]),
        );
        let id = *ids.entry(k).or_insert_with(|| {
            acc.push((Vec3::ZERO, Vec3::ZERO, [0.0; 2], 0));
            (acc.len() - 1) as u32
        });
        let a = &mut acc[id as usize];
        a.0 += *p;
        a.1 += m.nrm[i];
        a.2 = m.uv[i];
        a.3 += 1;
        map[i] = id;
    }
    let mut out = DaeMesh {
        name: m.name.clone(),
        geometry: m.geometry.clone(),
        materials: m.materials.clone(),
        ..Default::default()
    };
    for (p, n, uv, c) in acc {
        out.pos.push(p / c as f32);
        out.nrm.push(n.normalize_or(Vec3::Z));
        out.uv.push(uv);
    }
    let mut seen = std::collections::HashSet::new();
    for (t, tri) in m.tris.iter().enumerate() {
        let v = tri.map(|x| map[x as usize]);
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
            continue;
        }
        let mut key = v;
        key.sort_unstable();
        if seen.insert(key) {
            out.tris.push(v);
            out.tri_mat.push(m.tri_mat[t]);
        }
    }
    out
}

/// The distant level's cluster size (m), and the distance (m) it switches
/// at: where a cell is `LOD_PIXELS` pixels at 1080p with a 0.75 rad field of
/// view (a pixel is 2·d·tan(0.375)/1080 m at d). A cluster's vertex is its
/// vertices' mean, so on a smooth surface it moves far less than a cell. The
/// near level is always the full mesh.
pub const LOD_CELL: f32 = 0.03;
pub const LOD_PIXELS: f32 = 2.6;
/// The third (farthest) level's cluster size (m).
pub const FAR_CELL: f32 = 0.09;
pub fn lod_distance(cell: f32) -> f32 {
    cell / LOD_PIXELS * 1080.0 / (2.0 * 0.375f32.tan())
}

/// A Collada material name's class in Rubble's renderer (BeamNG's own
/// materials are textures; this is the shading the renderer can give them).
pub fn material_class(name: &str) -> u8 {
    let n = name.to_ascii_lowercase();
    let has = |k: &[&str]| k.iter().any(|s| n.contains(s));
    if has(&["windshield", "windscreen"]) {
        mat::SCREEN
    } else if has(&["glass", "window"]) {
        mat::GLASS
    } else if has(&["taillight", "tail_light", "brakelight", "lights_r", "red"]) {
        mat::TAIL_LENS
    } else if has(&["light", "lamp", "lens"]) {
        mat::LENS
    } else if has(&["chrome", "mirror"]) {
        mat::CHROME
    } else if has(&["tire", "tyre"]) {
        mat::TYRE
    } else if has(&["rim", "wheel", "hub"]) {
        mat::RIM
    } else if has(&["interior", "seat", "dash", "steer", "carpet", "gauge"]) {
        mat::SEAT
    } else if has(&["engine", "brake", "caliper", "exhaust", "suspension", "chassis", "frame", "metal"]) {
        mat::ENGINE
    } else if has(&["black", "trim", "plastic", "rubber", "grille", "grill", "plate", "under", "seal"]) {
        mat::TRIM
    } else {
        mat::PAINT
    }
}

/// What Rubble's renderer uses of a BeamNG material (`*.materials.json`):
/// its first stage's diffuse colour, and whether it is translucent, glows or
/// is textured (its colour map: `color_map`).
#[derive(Clone, Debug, Default)]
pub struct MatInfo {
    pub color: Option<[f32; 3]>,
    pub translucent: bool,
    pub emissive: bool,
    pub textured: bool,
    /// The colour map's file name (the path's last component: BeamNG paths
    /// are game-absolute, the files sit in the vehicle's folder).
    pub color_map: Option<String>,
}

/// Read a `*.materials.json` (relaxed JSON): every material by its `mapTo`
/// (the name meshes use) and its own name.
pub fn read_materials(src: &str) -> HashMap<String, MatInfo> {
    let mut out = HashMap::new();
    let Ok(Value::Object(m)) = crate::jbeam::parse_relaxed(src) else {
        return out;
    };
    for (key, v) in m {
        let st0 = v.get("Stages").and_then(|s| s.get(0));
        let f = |k: &str| st0.and_then(|s| s.get(k));
        let color = f("diffuseColor")
            .or_else(|| f("baseColorFactor"))
            .and_then(Value::as_array)
            .filter(|a| a.len() >= 3)
            .map(|a| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.5) as f32));
        let color_map = f("colorMap")
            .or_else(|| f("baseColorMap"))
            .and_then(Value::as_str)
            .map(|p| p.rsplit(['/', '\\']).next().unwrap_or(p).to_string());
        let textured = color_map.is_some();
        let translucent = v.get("translucent").and_then(Value::as_bool).unwrap_or(false)
            || v.get("translucentBlendOp").and_then(Value::as_str).is_some_and(|o| o != "None");
        let emissive = f("glow").and_then(Value::as_bool).unwrap_or(false)
            || f("emissive").and_then(Value::as_bool).unwrap_or(false)
            || f("emissiveFactor").is_some_and(|x| x.is_array());
        let info = MatInfo { color, translucent, emissive, textured, color_map };
        if let Some(t) = v.get("mapTo").and_then(Value::as_str) {
            out.insert(t.to_string(), info.clone());
        }
        out.insert(key, info);
    }
    out
}

fn vec3_of(v: Option<&Value>, d: Vec3) -> Vec3 {
    let f = |k: &str| v.and_then(|o| o.get(k)).and_then(Value::as_f64).map(|x| x as f32);
    match v {
        Some(Value::Object(_)) => Vec3::new(f("x").unwrap_or(d.x), f("y").unwrap_or(d.y), f("z").unwrap_or(d.z)),
        Some(Value::Array(a)) if a.len() == 3 => Vec3::new(
            a[0].as_f64().unwrap_or(d.x as f64) as f32,
            a[1].as_f64().unwrap_or(d.y as f64) as f32,
            a[2].as_f64().unwrap_or(d.z as f64) as f32,
        ),
        _ => d,
    }
}

/// What the visual was built from: counts and what was skipped.
#[derive(Clone, Debug, Default)]
pub struct FlexReport {
    pub flexbodies: usize,
    pub props: usize,
    pub vertices: usize,
    pub triangles: usize,
    /// Meshes a row names that no Collada file has, and vertices with no frame.
    pub missing: Vec<String>,
    pub unbound: usize,
}

/// The visual of a loaded JBeam vehicle: `flex` and `props` its assembled
/// `flexbodies` and `props` rows ([`crate::jbeam::assembled_rows`]), `desc`
/// the loaded vehicle, `sim` the solver it runs on (the visual binds to its
/// node indices: a desc node maps to the solver node at its rest position, so
/// either solver path works), `meshes` every Collada mesh.
///
/// Deform groups: a flexbody or prop row's `deformGroup` makes a damage unit
/// of the beams in that group (mapped to the solver's by their nodes), broken
/// once one breaks or yields past the group's `deformationTriggerRatio`, the
/// rule the solver's deform groups follow. A glass mesh then shatters
/// (`windshield` meshes crack instead), a lamp goes dark.
pub fn jbeam_visual(
    flex: &[(String, Map<String, Value>)],
    props: &[(String, Map<String, Value>)],
    desc: &SoftVehicleDesc,
    sim: &crate::explicit_beams::ExplicitBeams,
    meshes: &[DaeMesh],
    materials: &HashMap<String, MatInfo>,
    paint: [f32; 3],
) -> Result<(ShowcaseVisual, FlexReport), String> {
    // A mesh material's class and colour: its name's class, made glass when
    // the material is translucent and emissive when it glows; its diffuse
    // colour when it is untextured (a textured one shades a flat mid grey,
    // except paint, which takes the paint).
    // Textures by colour map, in first use order.
    let mut textures: Vec<String> = Vec::new();
    let mut shade = |name: &str| -> (u8, [f32; 3], u16) {
        let mi = materials.get(name).or_else(|| materials.get(name.trim_end_matches("-material"))).cloned().unwrap_or_default();
        let mut c = material_class(name);
        if mi.translucent && !matches!(c, mat::SCREEN | mat::GLASS | mat::LENS | mat::TAIL_LENS) {
            c = mat::GLASS;
        }
        if mi.emissive {
            c = if c == mat::TAIL_LENS { mat::TAIL_LED } else { mat::DRL };
        }
        // A colour map multiplies the material's colour (white unless the
        // material gives one); paint keeps the paint.
        let tex = match &mi.color_map {
            Some(f) if c != mat::PAINT => {
                let k = textures.iter().position(|t| t == f).unwrap_or_else(|| {
                    textures.push(f.clone());
                    textures.len() - 1
                });
                k as u16
            }
            _ => u16::MAX,
        };
        let col = match (mi.color, mi.textured) {
            (Some(k), false) if !matches!(c, mat::GLASS | mat::SCREEN) => k,
            (k, true) if c != mat::PAINT => k.unwrap_or([1.0, 1.0, 1.0]),
            _ => crate::authoring::skin::color(c, paint),
        };
        (c, col, tex)
    };
    let sim_nodes: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos).collect();
    let sim_nodes = &sim_nodes[..];
    let to_r = |p: Vec3| Vec3::from_array(crate::jbeam::to_rubble(p.to_array()));
    // Desc node → solver node (same rest position).
    let key = |p: Vec3| ((p.x * 500.0).round() as i32, (p.y * 500.0).round() as i32, (p.z * 500.0).round() as i32);
    let mut at: HashMap<(i32, i32, i32), u32> = HashMap::new();
    for (i, p) in sim_nodes.iter().enumerate() {
        at.entry(key(*p)).or_insert(i as u32);
    }
    let sim_of: Vec<Option<u32>> = desc
        .nodes
        .iter()
        .map(|n| {
            let p = Vec3::from_array(n.position);
            let k = key(p);
            (-1..=1)
                .flat_map(|dx| (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| (k.0 + dx, k.1 + dy, k.2 + dz))))
                .find_map(|k| at.get(&k).copied())
        })
        .collect();
    let by_name: HashMap<&str, u32> = desc
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.as_str(), i as u32))
        .collect();
    let mesh_by: HashMap<&str, &DaeMesh> = meshes
        .iter()
        .flat_map(|m| [(m.name.as_str(), m), (m.geometry.as_str(), m)])
        .collect();
    let mut out = ShowcaseVisual {
        node_count: sim_nodes.len() as u32,
        node_rest: sim_nodes.iter().map(|p| p.to_array()).collect(),
        ..Default::default()
    };
    let mut rep = FlexReport::default();
    let mut add = |name: &str,
                   m: &DaeMesh,
                   xf: Mat4,
                   pool: &[u32],
                   rigid: Option<[u32; 3]>,
                   sim_pool: Option<&[u32]>,
                   out: &mut ShowcaseVisual,
                   rep: &mut FlexReport| {
        let first = out.positions.len() as u32;
        let pos: Vec<Vec3> = m.pos.iter().map(|&p| to_r(xf.transform_point3(p))).collect();
        let nrm: Vec<Vec3> = m.nrm.iter().map(|&n| to_r(xf.transform_vector3(n)).normalize_or(Vec3::Y)).collect();
        let binds: Vec<Option<[u32; 3]>> = if let Some(t) = rigid {
            vec![Some(t); pos.len()]
        } else {
            let pool_sim: Vec<u32> = if let Some(sp) = sim_pool { sp.to_vec() } else { pool.iter().filter_map(|&i| sim_of[i as usize]).collect() };
            crate::flexbody::bind(&pos, sim_nodes, Some(&pool_sim), crate::flexbody::BindOptions::default())
                .into_iter()
                .map(|b| b.map(|b| b.nodes))
                .collect()
        };
        let mut vmap = vec![u32::MAX; pos.len()];
        for (k, b) in binds.iter().enumerate() {
            let Some(t) = b else {
                rep.unbound += 1;
                continue;
            };
            let (p0, p1, p2) = (sim_nodes[t[0] as usize], sim_nodes[t[1] as usize], sim_nodes[t[2] as usize]);
            let tt = (p1 - p0).normalize_or_zero();
            let nn = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            let bb = nn.cross(tt);
            let d = pos[k] - p0;
            let n = nrm[k];
            let (c, nl) = if rigid.is_some() {
                ([d.dot(tt), d.dot(bb), d.dot(nn)], [n.dot(tt), n.dot(bb), n.dot(nn)])
            } else {
                crate::authoring::skin::frame_coords(sim_nodes, *t, pos[k], n)
            };
            vmap[k] = out.positions.len() as u32;
            out.positions.push(pos[k].to_array());
            out.normals.push(n.to_array());
            out.uvs.push(m.uv.get(k).copied().unwrap_or([0.0; 2]));
            out.bind_nodes.push(*t);
            out.bind_coords.push(c);
            out.bind_normal.push(nl);
            out.bind_rest.push([p1.distance(p0), p2.distance(p0)]);
            out.ao.push(1.0);
            out.smooth.push(0);
            out.rigid.push(u8::from(rigid.is_some()));
            out.material.push(mat::PAINT);
            out.colors.push(paint);
            out.tex.push(u16::MAX);
        }
        for (ti, t) in m.tris.iter().enumerate() {
            let v = [vmap[t[0] as usize], vmap[t[1] as usize], vmap[t[2] as usize]];
            if v.contains(&u32::MAX) {
                continue;
            }
            let (mc, col, tx) = shade(m.materials.get(m.tri_mat[ti] as usize).map_or("", String::as_str));
            for &x in &v {
                out.material[x as usize] = mc;
                out.colors[x as usize] = col;
                out.tex[x as usize] = tx;
            }
            out.indices.extend_from_slice(&v);
        }
        let count = out.positions.len() as u32 - first;
        out.pieces.push((name.to_string(), first, count));
        rep.vertices += count as usize;
    };
    // Deform-group units, by name.
    let mut unit_of: HashMap<String, u32> = HashMap::new();
    let mut unit_kind_of = |name: &str, kind: u8, out: &mut ShowcaseVisual| -> u32 {
        if let Some(&u) = unit_of.get(name) {
            return u;
        }
        out.units.push(crate::authoring::skin::DamageUnit {
            name: name.to_string(),
            kind,
            beams: Vec::new(),
            strain: f32::INFINITY,
            shards: 0,
        });
        let u = (out.units.len() - 1) as u32;
        unit_of.insert(name.to_string(), u);
        u
    };
    let deform_of = |row: &Map<String, Value>| row.get("deformGroup").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
    for (pname, row) in flex {
        {
            let row = row.clone();
            let Some(mesh) = row.get("mesh").and_then(Value::as_str) else { continue };
            let Some(m) = mesh_by.get(mesh) else {
                rep.missing.push(mesh.to_string());
                continue;
            };
            let groups: Vec<String> = match row.get("[group]").or_else(|| row.get("group")) {
                Some(Value::Array(a)) => a.iter().filter_map(|g| g.as_str().map(str::to_string)).collect(),
                Some(Value::String(s)) => vec![s.clone()],
                _ => Vec::new(),
            };
            let mut pool: Vec<u32> = groups
                .iter()
                .filter_map(|g| desc.node_group_nodes.iter().find(|(n, _)| n == g).map(|x| &x.1).or_else(|| desc.part(g).map(|p| &p.nodes)))
                .flat_map(|p| p.iter().copied())
                .collect();
            pool.sort_unstable();
            pool.dedup();
            // A wheel's groups (`wheelhub_FR`, `tire_FR`) are the nodes its
            // pressureWheels row makes, which the solver appends after the
            // desc's: the wheel's own nodes near the mesh, and its hub.
            let wheelish = groups.iter().any(|g| ["wheel", "tire", "tyre", "hub"].iter().any(|k| g.to_ascii_lowercase().contains(k)));
            let mut wheel_pool: Vec<u32> = Vec::new();
            if pool.len() < 3 && wheelish && !m.pos.is_empty() {
                let c = m.pos.iter().map(|&p| to_r(p)).sum::<Vec3>() / m.pos.len() as f32;
                let suffix: Vec<String> = groups.iter().filter_map(|g| g.rsplit('_').next().map(str::to_string)).collect();
                for w in &desc.pressure_wheels {
                    if suffix.iter().any(|s| w.name.ends_with(s.as_str())) {
                        wheel_pool.extend([w.node1, w.node2].iter().filter_map(|&n| sim_of.get(n as usize).copied().flatten()));
                    }
                }
                wheel_pool.extend(
                    (desc.nodes.len()..sim_nodes.len())
                        .filter(|&i| sim_nodes[i].distance(c) < 0.6)
                        .map(|i| i as u32),
                );
            }
            if pool.len() < 3 && wheel_pool.len() < 3 {
                rep.missing.push(format!("{mesh}: groups {groups:?} have {} nodes", pool.len()));
                continue;
            }
            // pos (m), rot (degrees, about x, y, z), scale: in the JBeam frame.
            let t = vec3_of(row.get("pos"), Vec3::ZERO);
            let r = vec3_of(row.get("rot"), Vec3::ZERO);
            let s = vec3_of(row.get("scale"), Vec3::ONE);
            let xf = Mat4::from_scale_rotation_translation(
                s,
                Quat::from_euler(glam::EulerRot::XYZ, r.x.to_radians(), r.y.to_radians(), r.z.to_radians()),
                t,
            );
            let first = out.positions.len();
            add(&format!("{pname}:{mesh}"), m, xf, &pool, None, (pool.len() < 3).then_some(&wheel_pool[..]), &mut out, &mut rep);
            if let Some(g) = deform_of(&row) {
                let windshield = mesh.to_ascii_lowercase().contains("windshield") || mesh.to_ascii_lowercase().contains("windscreen");
                let kind = if windshield { crate::authoring::skin::unit_kind::LAMINATED } else { crate::authoring::skin::unit_kind::TEMPERED };
                let u = unit_kind_of(&g, kind, &mut out);
                out.unit.resize(out.positions.len(), u32::MAX);
                for k in first..out.positions.len() {
                    out.unit[k] = u;
                }
            }
            rep.flexbodies += 1;
        }
    }
    for (pname, row) in props {
        {
            let row = row.clone();
            let Some(mesh) = row.get("mesh").and_then(Value::as_str) else { continue };
            if mesh == "SPOTLIGHT" || mesh == "POINTLIGHT" {
                continue;
            }
            let Some(m) = mesh_by.get(mesh) else {
                rep.missing.push(mesh.to_string());
                continue;
            };
            let id = |k: &str| row.get(k).and_then(Value::as_str).and_then(|n| by_name.get(n).copied()).and_then(|i| sim_of[i as usize]);
            let (Some(a), Some(b), Some(c)) = (id("idRef"), id("idX"), id("idY")) else {
                rep.missing.push(format!("{mesh}: prop nodes"));
                continue;
            };
            let t = vec3_of(row.get("translation"), Vec3::ZERO);
            let r = vec3_of(row.get("baseRotation"), Vec3::ZERO);
            let xf = Mat4::from_rotation_translation(
                Quat::from_euler(glam::EulerRot::XYZ, r.x.to_radians(), r.y.to_radians(), r.z.to_radians()),
                t,
            );
            let first = out.positions.len();
            add(&format!("{pname}:prop:{mesh}"), m, xf, &[], Some([a, b, c]), None, &mut out, &mut rep);
            // A lamp prop's lit parts: emissive (dark once its deform group
            // has triggered).
            let func = row.get("func").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase();
            let lamp = ["light", "beam", "signal", "brake", "reverse", "fog"].iter().any(|k| func.contains(k));
            if lamp {
                let red = ["brake", "tail", "reverse"].iter().any(|k| func.contains(k));
                for k in first..out.positions.len() {
                    if matches!(out.material[k], mat::LENS | mat::TAIL_LENS | mat::PAINT | mat::CHROME) {
                        out.material[k] = if red { mat::TAIL_LED } else { mat::DRL };
                        out.colors[k] = crate::authoring::skin::color(out.material[k], paint);
                    }
                }
            }
            if let Some(g) = deform_of(&row) {
                let kind = if lamp { crate::authoring::skin::unit_kind::LAMP } else { crate::authoring::skin::unit_kind::MIRROR };
                let u = unit_kind_of(&g, kind, &mut out);
                out.unit.resize(out.positions.len(), u32::MAX);
                for k in first..out.positions.len() {
                    out.unit[k] = u;
                }
            }
            rep.props += 1;
        }
    }
    rep.triangles = out.indices.len() / 3;
    out.unit.resize(out.positions.len(), u32::MAX);
    // The units' beams: the desc beams in each deform group, as solver beams
    // (by node pair), and the group's trigger ratio.
    let mut by_pair: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (j, b) in sim.beams.iter().enumerate() {
        by_pair.entry((b.a.min(b.b), b.a.max(b.b))).or_default().push(j as u32);
    }
    for (i, b) in desc.beams.iter().enumerate() {
        let (deform, _) = desc.beam_group_names(i);
        for g in deform {
            let Some(&u) = unit_of.get(g) else { continue };
            let (Some(a), Some(c)) = (sim_of[b.a as usize], sim_of[b.b as usize]) else { continue };
            if let Some(js) = by_pair.get(&(a.min(c), a.max(c))) {
                let un = &mut out.units[u as usize];
                un.beams.extend(js.iter().copied());
                let r = b.jbeam.map_or(f32::INFINITY, |j| j.deformation_trigger_ratio);
                un.strain = un.strain.min(r);
            }
        }
    }
    for u in &mut out.units {
        u.beams.sort_unstable();
        u.beams.dedup();
    }
    // Shard seeds over the breaking units' glass and lenses.
    out.shard = vec![0; out.positions.len()];
    let mut acc = vec![0.0f32; out.units.len()];
    for t in out.indices.chunks(3) {
        let u = out.unit[t[0] as usize];
        if u == u32::MAX || out.units[u as usize].kind == crate::authoring::skin::unit_kind::LAMINATED {
            continue;
        }
        let p = |k: u32| Vec3::from_array(out.positions[k as usize]);
        acc[u as usize] += 0.5 * (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() * crate::authoring::skin::SHARD_DENSITY;
        while acc[u as usize] >= 1.0 {
            let Some(&v) = t.iter().find(|&&v| out.shard[v as usize] == 0) else { break };
            out.shard[v as usize] = 1;
            out.units[u as usize].shards += 1;
            acc[u as usize] -= 1.0;
        }
    }
    // Per node its solver beams (no tyre beams), the beams' spawn lengths,
    // node materials; no skin node normals (flexbodies shade by their frames).
    let nn = sim_nodes.len();
    let mut tyre = vec![false; nn];
    for t in &sim.tyres {
        for i in t.first..t.first + t.count {
            tyre[i as usize] = true;
        }
    }
    let mut count = vec![0u32; nn + 1];
    for b in &sim.beams {
        if !tyre[b.a as usize] && !tyre[b.b as usize] {
            count[b.a as usize + 1] += 1;
            count[b.b as usize + 1] += 1;
        }
    }
    for i in 0..nn {
        count[i + 1] += count[i];
    }
    let mut fill = count.clone();
    let mut list = vec![0u32; count[nn] as usize];
    for (j, b) in sim.beams.iter().enumerate() {
        if !tyre[b.a as usize] && !tyre[b.b as usize] {
            for n in [b.a, b.b] {
                list[fill[n as usize] as usize] = j as u32;
                fill[n as usize] += 1;
            }
        }
    }
    out.node_beam_start = count;
    out.node_beam_list = list;
    out.beam_rest0 = sim.beams.iter().map(|b| b.rest0).collect();
    out.node_tri_start = vec![0; nn + 1];
    out.node_rest_normal = vec![[0.0, 1.0, 0.0]; nn];
    out.node_material = tyre
        .iter()
        .map(|&t| if t { crate::authoring::skin::node_mat::RUBBER } else { crate::authoring::skin::node_mat::METAL })
        .collect();
    out.textures = textures;
    Ok((out, rep))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube of eight nodes in group `box` (JBeam frame), and a
    /// Collada quad (a polylist, under a translated scene node) as its
    /// flexbody: every vertex binds, and the rest pose reproduces the mesh.
    #[test]
    fn a_flexbody_binds_to_its_group_and_reproduces_its_rest_mesh() {
        let jbeam = r#"{ "toy": { "slotType": "main", "information": {"name": "toy"},
          "flexbodies": [ ["mesh", "[group]:", "nonFlexMaterials"], ["panel", ["box"]] ],
          "nodes": [ ["id", "posX", "posY", "posZ"], {"group": "box"},
            ["a", 0, 0, 0], ["b", 1, 0, 0], ["c", 0, 1, 0], ["d", 1, 1, 0],
            ["e", 0, 0, 1], ["f", 1, 0, 1], ["g", 0, 1, 1], ["h", 1, 1, 1] ],
          "beams": [ ["id1:", "id2:"],
            ["a","b"],["a","c"],["b","d"],["c","d"],["e","f"],["e","g"],["f","h"],["g","h"],
            ["a","e"],["b","f"],["c","g"],["d","h"],["a","d"],["e","h"],["a","f"],["c","h"],["a","g"],["b","h"] ] } }"#;
        let dae = r##"<?xml version="1.0"?><COLLADA><asset><up_axis>Z_UP</up_axis></asset>
          <library_geometries><geometry id="g" name="panel"><mesh>
            <source id="p"><float_array count="12">0 0 0 0.8 0 0 0.8 0.8 0 0 0.8 0</float_array>
              <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
            <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
            <polylist material="m" count="1"><input semantic="VERTEX" source="#v" offset="0"/>
              <vcount>4</vcount><p>0 1 2 3</p></polylist>
          </mesh></geometry></library_geometries>
          <library_visual_scenes><visual_scene id="s"><node name="panel"><translate>0.1 0.1 0.5</translate>
            <instance_geometry url="#g"/></node></visual_scene></library_visual_scenes></COLLADA>"##;
        let meshes = read_collada(dae).unwrap();
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].tris.len(), 2, "a quad fans into two triangles");
        assert!((meshes[0].pos[2] - Vec3::new(0.9, 0.9, 0.5)).length() < 1e-6, "the node's translation");
        let files = [("toy.jbeam", jbeam)];
        let desc = crate::jbeam::load_jbeam(&files, None).unwrap();
        let flex = crate::jbeam::assembled_rows(&files, None, "flexbodies").unwrap();
        let sim = crate::explicit_beams::ExplicitBeams::new(&desc, Vec3::ZERO, Vec3::ZERO, Default::default());
        let (v, rep) = jbeam_visual(&flex, &[], &desc, &sim, &meshes, &HashMap::new(), [0.5; 3]).unwrap();
        assert_eq!((rep.flexbodies, rep.unbound, rep.triangles), (1, 0, 2), "{rep:?}");
        let nodes: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos).collect();
        let (p, _) = crate::authoring::skin::deform(&v, &nodes);
        for (a, b) in p.iter().zip(&v.positions) {
            assert!(a.distance(Vec3::from_array(*b)) < 1e-5);
        }
        // Rubble's frame: JBeam z up is Rubble's y.
        assert!((v.positions[0][1] - 0.5).abs() < 1e-6);
    }
}
