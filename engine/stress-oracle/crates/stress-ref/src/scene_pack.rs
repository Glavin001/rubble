//! Import of the repository's authored destruction assets ("scene packs", the JSON
//! written by `blast/blast-stress-solver/structures`) as stress-oracle scenes.
//!
//! Geometry and structure come from the asset: chunk shapes (boxes and convex hulls),
//! positions, which chunks are anchored, structural roles and the bond graph. Physics
//! comes from this crate: each asset material is mapped to one of the calibrated
//! materials in `material.rs`. The assets' own stress limits are tuned for gameplay
//! (their "reinforced concrete" breaks in tension at 11.5 MPa, about four times real
//! concrete) and are not used.
//!
//! Conversions:
//! * Packs are Y-up; scenes are Z-up: `(x, y, z) -> (x, -z, y)` (a rotation, so shapes
//!   keep their handedness).
//! * Hull points are given relative to the node centroid; the chunk centre is moved to
//!   the hull's centre of mass (exactly computed) and the points re-centred on it.
//! * A node with zero mass is a support: the chunk is fixed.
//! * The ground is the assets' grade, `y = 0` (the demo's ground plane); their footings
//!   are embedded below it. A chunk reaching down to grade bears on the soil, which the
//!   oracle does not model: it is a support, as the bottom courses of the oracle's own
//!   scenes are.
//! * Joints of reinforced concrete carry the code-minimum reinforcement (EN 1992-1-1
//!   9.2.1.1: `As,min = max(0.26 fctm / fyk, 0.0013)` of the section), from the
//!   calibrated concrete and steel; the assets give no reinforcement.
//! * Bond patches are recomputed from the touching faces of the two chunks (area,
//!   centroid, principal widths); the asset's centroid, normal and area are the fallback
//!   where no shared face is found within the seam tolerance.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::builders::{body, chunk, new_scene};
use crate::hull::{contact_patch, ConvexHull, MAX_HULL_VERTICES};
use crate::material::Material;
use crate::math::Vec3;
use crate::scene::{BondDesc, ChunkDesc, GroundDesc, RebarSpec, Scene, Support};

/// Faces closer than this (m) count as touching: the assets' Voronoi cutter leaves
/// hairline seams between neighbouring cells.
const SEAM: f64 = 5e-3;

/// The assets' ground plane (Z-up height).
const GRADE: f64 = 0.0;

/// Asset materials that are reinforced concrete.
fn is_reinforced_concrete(asset: &str) -> bool {
    matches!(asset, "reinforced-concrete" | "prestressed-concrete" | "concrete-slab" | "footing-anchor")
}

/// Code-minimum reinforcement ratio of a section (EN 1992-1-1 9.2.1.1).
pub(crate) fn minimum_reinforcement(concrete: &Material, steel: &Material) -> f64 {
    (0.26 * concrete.tensile_strength / steel.tensile_strength).max(0.0013)
}

/// The calibrated material standing for an asset material, as (chunk, joint) material
/// names: masonry units are joined by mortar.
fn map_material(asset: &str) -> Result<(&'static str, &'static str), String> {
    Ok(match asset {
        // Precast facade panels are concrete too.
        "reinforced-concrete" | "prestressed-concrete" | "concrete-slab" | "footing-anchor" | "facade-panel" => ("concrete", "concrete"),
        "brick" => ("brick", "mortar"),
        "stone" | "white-stone" => ("stone", "mortar"),
        "glass" => ("glass", "glass"),
        // Facade and glazing clips are metal connectors.
        "steel" | "facade-clip" | "glazing-clip" => ("steel", "steel"),
        "wood-frame" => ("timber", "timber"),
        other => return Err(format!("no calibrated material for asset material '{other}'")),
    })
}

pub(crate) fn material_preset(name: &str) -> Material {
    match name {
        "concrete" => Material::concrete(),
        "brick" => Material::brick(),
        "mortar" => Material::mortar(),
        "stone" => Material::stone(),
        "glass" => Material::glass(),
        "steel" => Material::steel(),
        "timber" => Material::timber(),
        _ => unreachable!("mapped names only"),
    }
}

/// Y-up to Z-up.
/// The calibrated base (density, stiffness, fracture energy) of an asset material in an
/// authored-strengths import, whose strengths come from the pack: the calibrated mapping
/// where there is one, else by the name's material class. A name of no known class fails.
fn map_material_authored(asset: &str) -> Result<(&'static str, &'static str), String> {
    if let Ok(m) = map_material(asset) {
        return Ok(m);
    }
    let has = |keys: &[&str]| keys.iter().any(|k| asset.contains(k));
    Ok(if has(&["glass", "glazing"]) {
        ("glass", "glass")
    } else if has(&["steel", "metal", "hydrant", "clip", "fastener"]) {
        ("steel", "steel")
    } else if has(&["mortar"]) {
        ("brick", "mortar")
    } else if has(&["brick"]) {
        ("brick", "mortar")
    } else if has(&["block"]) {
        // Blockwork (concrete masonry units in mortar).
        ("brick", "mortar")
    } else if has(&["stone", "masonry", "slate", "paving", "porcelain", "road"]) {
        ("stone", "mortar")
    } else if has(&["block"]) {
        // Concrete blockwork (blast-stress-solver's `block-wall`): concrete units in mortar.
        ("concrete", "mortar")
    } else if has(&["concrete", "footing", "asphalt", "slab", "facade", "plaster", "drywall", "gypsum"]) {
        ("concrete", "concrete")
    } else if has(&["wood", "timber", "oak", "joinery", "siding", "trim", "fibre", "poster", "paper", "ink", "upholstery", "bedding", "rubber", "plastic", "appliance", "furniture"]) {
        ("timber", "timber")
    } else {
        return Err(format!("no material class for asset material '{asset}' (authored strengths)"));
    })
}

/// Each pack material's crush block (ScenePack v3 `materials[].crush`, camelCase, SI),
/// by material index. A crush block in a pack declaring version 1 or 2 is an error (the
/// author expects behaviour the declared version does not have); a missing required
/// field or a non-positive one is an error naming the material. Strain-rate hardening
/// (`strainRateExponent`) is not modelled: a pack that asks for it gets a note.
fn pack_crush(json: &Value, notes: &mut Vec<String>) -> Result<Vec<Option<crate::crush::CrushParams>>, String> {
    let version = json["version"].as_u64().unwrap_or(1);
    let Some(materials) = json["defaults"]["solver"]["materials"].as_array() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(materials.len());
    for (i, m) in materials.iter().enumerate() {
        let name = m["name"].as_str().unwrap_or("").to_string();
        let Some(c) = m.get("crush").filter(|c| !c.is_null()) else {
            out.push(None);
            continue;
        };
        if version < 3 {
            return Err(format!("pack material {i} ('{name}') has a crush block but the pack declares version {version}; crushing is ScenePack v3"));
        }
        let required = |k: &str| c[k].as_f64().ok_or_else(|| format!("pack material {i} ('{name}'): crush.{k} is required"));
        let params = crate::crush::CrushParams {
            cap_pressure: required("capPressure")?,
            cohesion: c["cohesion"].as_f64().unwrap_or(0.0),
            friction_slope: c["frictionSlope"].as_f64().unwrap_or(0.0),
            crush_energy: required("crushEnergy")?,
            crush_viscosity: required("crushViscosity")?,
        };
        params.validate(&name)?;
        if c["strainRateExponent"].as_f64().is_some_and(|e| e > 0.0) {
            notes.push(format!("material '{name}': crush strain-rate hardening is not modelled"));
        }
        out.push(Some(params));
    }
    Ok(out)
}

fn zup(v: &Value) -> Vec3 {
    let f = |k: &str| v[k].as_f64().unwrap_or(0.0);
    Vec3::new(f("x"), -f("z"), f("y"))
}

/// A vector given as `{x, y, z}` or `[x, y, z]`, to Z-up.
fn vector(v: &Value) -> Vec3 {
    match v.as_array() {
        Some(a) => points(v).first().copied().filter(|_| a.len() == 3).unwrap_or(Vec3::ZERO),
        None => zup(v),
    }
}

/// A flat `[x, y, z, x, y, z, ...]` list (or a list of `{x, y, z}`) to Z-up points.
fn points(v: &Value) -> Vec<Vec3> {
    let arr = v.as_array().cloned().unwrap_or_default();
    if arr.first().is_some_and(Value::is_number) {
        arr.chunks(3)
            .map(|c| {
                let g = |i: usize| c[i].as_f64().unwrap_or(0.0);
                Vec3::new(g(0), -g(2), g(1))
            })
            .collect()
    } else {
        arr.iter().map(zup).collect()
    }
}

/// Load the scene pack at `path` as a scene named `name` with one body `body_name`,
/// on the assets' ground plane.
/// How a pack is imported.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImportOptions {
    /// Each asset material's joints take the pack's own elastic limits as their
    /// strengths (tension, compression, shear: what the pack's Blast stress solver
    /// breaks against), on the calibrated material's stiffness and fracture energy.
    /// For like-for-like runs against the game; off: the calibrated strengths.
    pub authored_strengths: bool,
    /// With `authored_strengths`: the pack's fatal limits instead of its elastic ones.
    /// Blast's damage law breaks a bond outright at its fatal limit and only erodes it,
    /// at a rate, between the two; on impacts and loads of a few frames it breaks there.
    /// For the like-for-like comparison with Blast; off: the elastic limits.
    pub authored_fatal: bool,
    /// The pack's own node masses (each chunk at the density that gives it) and its
    /// supports (nodes of mass 0 only), as its Blast solver sees them, instead of the
    /// calibrated densities and supports at grade. For the like-for-like comparison.
    pub pack_masses: bool,
    /// With `authored_strengths`: each joint material's Young's modulus is the pack's own
    /// (`elasticModulus`, scaled), the one its Blast solver weights bonds by, instead of
    /// the calibrated material's. For the like-for-like comparison.
    pub pack_modulus: bool,
    /// Also keep what Blast's own solver reads from the pack (`Scene::blast`): its node
    /// centroids, masses and volumes, its bond centroids, normals and areas, and every
    /// material's elastic and fatal limits. Every pack bond of positive area is kept, one to
    /// one with the scene's bonds (a bond whose shared face has no width gets a square
    /// patch of its area rather than being left out). For the Blast-model profiles.
    pub blast_authoring: bool,
}

pub fn import(path: &std::path::Path, name: &str, body_name: &str) -> Result<Scene, String> {
    import_with(path, name, body_name, ImportOptions::default())
}

/// `import` with options.
pub fn import_with(path: &std::path::Path, name: &str, body_name: &str, options: ImportOptions) -> Result<Scene, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let json: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    // A Shardwright export (`shardwright.rs`) is imported by its own reader.
    if json["format"] == "shardwright-stress/1" {
        return crate::shardwright::import(path, name);
    }
    let source = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    let (mut scene, notes) = import_value_with(&json, name, body_name, &source, options)?;
    for note in notes {
        scene.description = format!("{} ({note})", scene.description);
    }
    Ok(scene)
}

/// [`import_with`] from an already parsed pack (e.g. one fetched by a web page), named
/// `source` in the description. Also returns what the import could not take from the pack
/// as given (bond patches from the asset's own area, bonds left out and why), for the
/// caller to show.
pub fn import_value_with(json: &Value, name: &str, body_name: &str, source: &str, options: ImportOptions) -> Result<(Scene, Vec<String>), String> {
    let sc = &json["scenario"];
    let nodes = sc["nodes"].as_array().ok_or("pack has no nodes")?;
    let colliders = sc["nodeColliders"].as_array().ok_or("pack has no nodeColliders")?;
    let library = sc["shapeLibrary"].as_array().cloned().unwrap_or_default();
    // Version-1 packs (the pre-fractured Voronoi walls, bridges, towers) carry no
    // materials: they are concrete.
    let node_materials: Vec<String> = match sc["nodeMaterials"].as_array() {
        Some(a) => a.iter().map(|m| m.as_str().unwrap_or("").to_string()).collect(),
        None => vec!["reinforced-concrete".to_string(); nodes.len()],
    };
    let node_types: Vec<String> = sc["nodeTypes"].as_array().map(|a| a.iter().map(|t| t.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
    // The building each node belongs to (`house@house-18`), for selecting it (targets).
    let node_groups: Vec<String> = sc["nodeGroups"].as_array().map(|a| a.iter().map(|t| t.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
    let pack_materials: Vec<String> = json["defaults"]["solver"]["materials"]
        .as_array()
        .map(|a| a.iter().map(|m| m["name"].as_str().unwrap_or("").to_string()).collect())
        .unwrap_or_default();
    // The pack's own limits by material index (names repeat with different limits), Pa,
    // scaled by its `materialScale`.
    let scale = json["defaults"]["solver"]["materialScale"].as_f64().unwrap_or(1.0);
    let authored: Vec<Option<(f64, f64, f64)>> = json["defaults"]["solver"]["materials"]
        .as_array()
        .map(|a| {
            let (t, c, sh) = if options.authored_fatal { ("tensionFatal", "compressionFatal", "shearFatal") } else { ("tensionElastic", "compressionElastic", "shearElastic") };
            a.iter().map(|m| Some((m[t].as_f64()? * scale, m[c].as_f64()? * scale, m[sh].as_f64()? * scale))).collect()
        })
        .unwrap_or_default();
    let pack_moduli: Vec<Option<f64>> = json["defaults"]["solver"]["materials"].as_array().map(|a| a.iter().map(|m| Some(m["elasticModulus"].as_f64()? * scale)).collect()).unwrap_or_default();
    // ScenePack v3: each pack material's optional crush block (chunk crushing); a node's
    // own material (`nodes[].m`) selects it.
    let mut notes = Vec::new();
    let crush = pack_crush(json, &mut notes)?;
    let classify = |asset: &str| if options.authored_strengths { map_material_authored(asset) } else { map_material(asset) };
    let mut blast = options.blast_authoring.then(|| crate::blast_bond::BlastAuthoring {
        materials: json["defaults"]["solver"]["materials"]
            .as_array()
            .map(|a| {
                let f = |m: &Value, k: &str| m[k].as_f64().unwrap_or(0.0) * scale;
                a.iter()
                    .map(|m| crate::blast_bond::BlastMaterial {
                        compression_elastic: f(m, "compressionElastic"),
                        compression_fatal: f(m, "compressionFatal"),
                        tension_elastic: f(m, "tensionElastic"),
                        tension_fatal: f(m, "tensionFatal"),
                        shear_elastic: f(m, "shearElastic"),
                        shear_fatal: f(m, "shearFatal"),
                        residual_area_fraction: m["residualAreaFraction"].as_f64().unwrap_or(0.0),
                        modulus: f(m, "elasticModulus"),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        ..Default::default()
    });

    let mut scene = new_scene(name, &format!("Imported from {source}"));
    let mut chunks: Vec<ChunkDesc> = Vec::with_capacity(nodes.len());
    let mut hulls: Vec<ConvexHull> = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let (mat, _) = classify(&node_materials[i])?;
        let centroid = zup(&node["centroid"]);
        let collider = &colliders[i];
        let collider = if collider["kind"] == "shape" { &library[collider["shape"].as_u64().ok_or("bad shape index")? as usize] } else { collider };
        let local: Vec<Vec3> = match collider["kind"].as_str() {
            Some("cuboid") => {
                let h = vector(&collider["halfExtents"]).abs();
                (0..8).map(|k| Vec3::new(if k & 1 == 0 { -h.x } else { h.x }, if k & 2 == 0 { -h.y } else { h.y }, if k & 4 == 0 { -h.z } else { h.z })).collect()
            }
            Some("convex_hull") => points(&collider["points"]),
            other => return Err(format!("node {i}: unsupported collider {other:?}")),
        };
        let mut hull = ConvexHull::new(&local).map_err(|e| format!("node {i}: {e}"))?;
        if hull.vertices.len() > MAX_HULL_VERTICES {
            return Err(format!("node {i}: hull has {} vertices", hull.vertices.len()));
        }
        let com = hull.unit_mass().centroid;
        hull = hull.translated(com);
        let half = hull.half_extents() * (1.0 + 1e-9);
        let mut c = chunk(centroid + com, half, mat);
        c.hull = Some(hull.vertices.iter().map(|v| v.to_array()).collect());
        let lowest = hull.vertices.iter().map(|v| v.z).fold(f64::INFINITY, f64::min) + c.center[2];
        let node_mass = node["mass"].as_f64().unwrap_or(0.0);
        if let Some(blast) = &mut blast {
            blast.nodes.push(crate::blast_bond::BlastNodeDesc { position: centroid.to_array(), mass: node_mass, volume: node["volume"].as_f64().unwrap_or(0.0) });
        }
        if node_mass <= 0.0 || (!options.pack_masses && lowest <= GRADE + SEAM) {
            c.support = Support::Fixed;
        }
        // The node's own material's crush block (v3), if it has one.
        // (Checked only in a pack that crushes: v2 packs keep importing as they did.)
        let node_crush = match node.get("m").and_then(Value::as_u64) {
            _ if crush.iter().all(Option::is_none) => None,
            Some(m) => *crush.get(m as usize).ok_or_else(|| format!("node {i}: material index {m} out of range ({} materials)", crush.len()))?,
            None => crush.first().copied().flatten(),
        };
        if options.pack_masses && node_mass > 0.0 {
            // A chunk material of its own, at the density that gives the pack's mass.
            let volume = hull.unit_mass().volume;
            let key = format!("{mat}#mass{i}");
            let mut m = material_preset(mat);
            m.density = node_mass / volume;
            m.crush = node_crush;
            scene.materials.insert(key.clone(), m);
            c.material = key;
        } else if let Some(cp) = node_crush.filter(|_| node_mass > 0.0) {
            let index = node.get("m").and_then(Value::as_u64).unwrap_or(0);
            let key = format!("{mat}@crush#{index}");
            scene.materials.entry(key.clone()).or_insert_with(|| Material { crush: Some(cp), ..material_preset(mat) });
            c.material = key;
        }
        if let Some(t) = node_types.get(i).filter(|t| !t.is_empty()) {
            c.groups.push(t.clone());
        }
        if let Some(g) = node_groups.get(i).filter(|g| !g.is_empty()) {
            c.groups.push(g.clone());
        }
        scene.materials.entry(mat.to_string()).or_insert_with(|| material_preset(mat));
        chunks.push(c);
        hulls.push(hull);
    }

    let mut bonds = Vec::new();
    let mut fallback = 0usize;
    let mut degenerate = 0usize;
    // Bonds the pack gives no normal: taken from the shared face, along the chunk centres.
    let mut unnormal = [0usize; 2];
    for b in sc["bonds"].as_array().ok_or("pack has no bonds")? {
        let (a, bb) = (b["node0"].as_u64().ok_or("bad bond")? as usize, b["node1"].as_u64().ok_or("bad bond")? as usize);
        let asset_material = b["m"].as_u64().and_then(|m| pack_materials.get(m as usize)).cloned().unwrap_or_else(|| node_materials[a].clone());
        let (_, calibrated) = classify(&asset_material)?;
        let index = b["m"].as_u64().map(|m| m as usize);
        let joint = match index.and_then(|m| authored.get(m).copied().flatten()).filter(|_| options.authored_strengths) {
            Some((tension, compression, shear)) => {
                let name = format!("{calibrated}@{asset_material}#{}", index.unwrap_or(0));
                scene.materials.entry(name.clone()).or_insert_with(|| {
                    let mut m = material_preset(calibrated);
                    m.tensile_strength = tension;
                    m.compressive_strength = compression;
                    m.cohesion = shear;
                    m.shear_cap = None;
                    if let Some(e) = index.and_then(|i| pack_moduli.get(i).copied().flatten()).filter(|_| options.pack_modulus) {
                        m.youngs_modulus = e;
                    }
                    m
                });
                name
            }
            None => {
                scene.materials.entry(calibrated.to_string()).or_insert_with(|| material_preset(calibrated));
                calibrated.to_string()
            }
        };
        // Normal from a to b (the asset's may point either way).
        let mut normal = zup(&b["normal"]).normalized();
        let (ca, cb) = (Vec3::from_array(chunks[a].center), Vec3::from_array(chunks[bb].center));
        if normal == Vec3::ZERO {
            // The pack gives no normal: that of the face of `a` meeting `b` over the
            // largest patch, else the line between the chunk centres.
            let shared = hulls[a]
                .faces
                .iter()
                .filter(|f| f.normal.dot(cb - ca) > 0.0)
                .filter_map(|f| contact_patch(&hulls[a], ca, &hulls[bb], cb, f.normal, SEAM).map(|p| (f.normal, p.area)))
                .max_by(|x, y| x.1.total_cmp(&y.1));
            match shared {
                Some((n, _)) => {
                    unnormal[0] += 1;
                    normal = n;
                }
                None => {
                    unnormal[1] += 1;
                    normal = (cb - ca).normalized();
                }
            }
        }
        if normal.dot(cb - ca) < 0.0 {
            normal = -normal;
        }
        // Under Blast authoring the pack decides which bonds exist: every one of positive area.
        let pack_area = b["area"].as_f64().unwrap_or(0.0);
        if blast.is_some() && !(pack_area > 0.0) {
            continue;
        }
        let patch = contact_patch(&hulls[a], ca, &hulls[bb], cb, normal, SEAM);
        let (centroid, mut area, tangent, mut width) = match patch {
            Some(p) => (p.centroid, p.area, p.tangent, p.width),
            None => {
                fallback += 1;
                let area = b["area"].as_f64().unwrap_or(0.0);
                (zup(&b["centroid"]), area, normal.any_perpendicular().normalized(), [area.sqrt(); 2])
            }
        };
        if blast.is_some() && !(area > 0.0 && width[0] > 0.0 && width[1] > 0.0) {
            degenerate += 1;
            if !(area > 0.0) {
                area = pack_area;
            }
            width = [area.sqrt(); 2];
        }
        if area <= 0.0 {
            continue;
        }
        // A patch with no width in one direction (two faces meeting along an edge, or a
        // sliver of shared face) has no section to bend about: not a bond.
        if !(width[0] > 0.0 && width[1] > 0.0) {
            degenerate += 1;
            continue;
        }
        if let Some(blast) = &mut blast {
            let material = b["m"].as_u64().map(|m| m as usize).unwrap_or(0);
            blast.bonds.push(crate::blast_bond::BlastBondDesc { centroid: zup(&b["centroid"]).to_array(), normal: zup(&b["normal"]).to_array(), area: pack_area, material });
        }
        let rebar = is_reinforced_concrete(&asset_material).then(|| {
            scene.materials.entry("steel".into()).or_insert_with(Material::steel);
            RebarSpec { area: area * minimum_reinforcement(&Material::concrete(), &Material::steel()), material: "steel".into() }
        });
        bonds.push(BondDesc {
            a,
            b: bb,
            centroid: centroid.to_array(),
            normal: normal.to_array(),
            tangent: tangent.to_array(),
            area,
            width,
            material: joint,
            rebar,
            buckling_length: None,
            level: 0,
            groups: Vec::new(),
            derived: None,
        });
    }
    if fallback > 0 {
        notes.push(format!("{fallback} of {} bond patches from the asset's area, no shared face found", bonds.len()));
    }
    if degenerate > 0 && blast.is_some() {
        notes.push(format!("{degenerate} asset bonds have no shared face of width: square patches of their area"));
    } else if degenerate > 0 {
        notes.push(format!("{degenerate} asset bonds left out: their shared face has no width"));
    }
    if unnormal != [0, 0] {
        notes.push(format!("{} asset bonds have no normal: {} taken from the shared face, {} along the chunk centres", unnormal[0] + unnormal[1], unnormal[0], unnormal[1]));
    }

    scene.materials.entry("concrete".into()).or_insert_with(Material::concrete);
    scene.ground = Some(GroundDesc { height: GRADE, friction: 0.6, material: "concrete".into() });
    scene.bodies.push(body(body_name, chunks, bonds));
    scene.blast = blast;
    scene.validate()?;
    Ok((scene, notes))
}

/// Bonds per asset material pair and chunk counts per calibrated material.
pub fn summary(scene: &Scene) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for b in &scene.bodies {
        for c in &b.chunks {
            *out.entry(format!("chunks:{}", c.material)).or_default() += 1;
        }
        for bd in &b.bonds {
            *out.entry(format!("bonds:{}", bd.material)).or_default() += 1;
        }
    }
    out
}
