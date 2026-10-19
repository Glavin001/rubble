//! Import of Shardwright pre-fractured assets (exported from `.fracphys` as
//! `shardwright-stress/1` JSON: fragments of levels 0..=L with their parents, and the
//! bonds of level L).
//!
//! The level-L fragments become chunks. Shardwright fragments are compounds of up to
//! 16 non-overlapping convex hulls (non-convex chunks), and each becomes a compound
//! chunk (`compound_chunk`) of its hulls, faces and all, as the asset gives them: centred on the hulls' centroid, along their
//! principal axes, with their mass properties at the calibrated material's density.
//! Contact is per hull, so neighbours do not overlap at rest. Compound chunks take
//! penalty contact: the scene has `layer_contact` off. Exports without per-hull points
//! (`hulls`) fall back to the equivalent-inertia box (`equivalent_box`), whose boxes
//! overlap their neighbours (median 82 mm on the two-storey building's bonded pairs).
//! Each chunk carries its ancestry as groups: `member:<L0 id>` (the authored
//! component), `name:<component name>` when the export has the bake's names (e.g.
//! `name:f0_col_21`, the selector of frac-sim's `--remove`), `l1:<id>`, and its role (`column`, `beam`, `slab`, `wall`, ...). Bonds to
//! the world (`b = -1`) make their chunk a support.
//!
//! Bond sections: the asset gives the area and the second moments `i_uu`, `i_vv` in its
//! `(u, v)` frame; the joint takes a rectangle of the same area and the aspect of the
//! patch's oriented extent, along `u` (an approximation of the bending stiffness of a
//! non-rectangular patch). Materials map to the calibrated presets; mortar joints
//! (`composition` kind 1) take mortar; reinforced concrete takes the asset's rebar, or
//! the EN 1992 minimum where the asset has none. The assets' strength scales (Weibull
//! and deliberate weakening of cut interfaces) are not used: strengths are the
//! calibrated materials'.
//!
//! Units SI; the asset's +Y up becomes +Z up.

use serde_json::Value;

use crate::builders::{body, chunk, new_scene};
use crate::hull::ConvexHull;
use crate::material::Material;
use crate::math::{Mat3, Vec3};
use crate::observation::Observation;
use crate::scene::{BondDesc, ChunkSelector, EventDesc, GroundDesc, ImpactorDesc, ImpactorShape, PartDesc, RebarSpec, Scene, Support};
use crate::world::World;
use crate::scene_pack::{material_preset, minimum_reinforcement};

const ROLES: [&str; 9] = ["generic", "column", "beam", "slab", "wall", "connection", "cosmetic", "prop", "glazing"];

/// The calibrated (chunk, joint) materials of a Shardwright library material.
fn map_material(name: &str) -> (&'static str, &'static str) {
    match name {
        "brick_clay" | "ceramic" => ("brick", "mortar"),
        "sandstone" => ("stone", "mortar"),
        "pine" => ("timber", "timber"),
        "glass_annealed" | "glass_tempered" => ("glass", "glass"),
        "steel_s355" => ("steel", "steel"),
        // Gypsum board: weak, brittle and light; mortar is the closest preset.
        "drywall" => ("mortar", "mortar"),
        _ => ("concrete", "concrete"),
    }
}

/// Y-up `[x, y, z]` to Z-up.
fn zup(v: &Value) -> Vec3 {
    let f = |i: usize| v[i].as_f64().unwrap_or(0.0);
    Vec3::new(f(0), -f(2), f(1))
}

/// Eigen-decomposition of a symmetric 3 x 3 matrix (cyclic Jacobi): eigenvalues and
/// eigenvectors as columns.
fn sym_eigen(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off <= 1e-15 * (a[0][0].abs() + a[1][1].abs() + a[2][2].abs()) {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Quaternion (w, x, y, z) of a proper rotation matrix.
fn quat_wxyz(r: [[f64; 3]; 3]) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [(r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [(r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [(r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s]
    };
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    q.map(|x| x / n)
}

/// A box chunk with the fragment's centre of mass, principal axes and principal moments
/// per unit mass (the uniform box whose inertia ellipsoid is the fragment's: `a^2 =
/// 6 (I2 + I3 - I1) / m`). A compound (multi-hull, non-convex) fragment's mass
/// distribution is kept without a convex hull of it; its mass is then the calibrated
/// density times the box's volume.
fn equivalent_box(f: &Value, mat: &str) -> Result<crate::scene::ChunkDesc, String> {
    let m = f["mass"].as_f64().filter(|m| *m > 0.0).ok_or("no mass")?;
    let i: Vec<f64> = f["inertia"].as_array().ok_or("no inertia")?.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
    // Y-up to Z-up: R (x, y, z) = (x, -z, y); I' = R I R^T.
    let r = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
    let ia: [[f64; 3]; 3] = std::array::from_fn(|p| std::array::from_fn(|q| i[3 * p + q]));
    let iz: [[f64; 3]; 3] = std::array::from_fn(|p| std::array::from_fn(|q| (0..3).map(|a| (0..3).map(|b| r[p][a] * ia[a][b] * r[q][b]).sum::<f64>()).sum()));
    let (l, mut v) = sym_eigen(iz);
    // A proper rotation.
    let det = v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) - v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0]) + v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]);
    if det < 0.0 {
        for row in v.iter_mut() {
            row[2] = -row[2];
        }
    }
    let side = |k: usize| -> f64 {
        let (a, b, c) = (l[k], l[(k + 1) % 3], l[(k + 2) % 3]);
        (6.0 * (b + c - a) / m).max(1e-6 * (a + b + c) / m).sqrt()
    };
    let half = Vec3::new(side(0), side(1), side(2)) * 0.5;
    let mut c = chunk(zup(&f["com"]), half, mat);
    c.orientation = quat_wxyz(v);
    Ok(c)
}

/// A compound chunk of the fragment's convex hulls (`hulls`, asset frame: each its
/// `points` and polygon `faces`, taken as given; older exports carry points only):
/// centred on their centroid and turned to their principal axes (proper rotations, so
/// the faces keep their winding), each hull's points in that frame.
fn compound_chunk(f: &Value, mat: &str) -> Result<crate::scene::ChunkDesc, String> {
    let hulls = f["hulls"].as_array().filter(|h| !h.is_empty()).ok_or("no hulls")?;
    let points = |h: &Value| -> Vec<Vec3> { h.get("points").unwrap_or(h).as_array().map(|p| p.iter().map(zup).collect()).unwrap_or_default() };
    let faces = |h: &Value| -> Option<Vec<Vec<usize>>> {
        h.get("faces")?.as_array()?.iter().map(|f| f.as_array().map(|i| i.iter().filter_map(|k| k.as_u64().map(|k| k as usize)).collect())).collect()
    };
    let parts: Vec<(Vec<Vec3>, Option<Vec<Vec<usize>>>)> = hulls.iter().map(|h| (points(h), faces(h))).collect();
    // A hull whose stored faces do not close (an edge used once, e.g. a face split
    // along a T-junction) is the convex hull of its own points: the same solid, with
    // faces recomputed. Only points that do not span a solid are an error.
    // Points that span no solid at all (a flat sheet) are left out like the zero-volume
    // hulls below.
    let parts: Vec<(Vec<Vec3>, Option<Vec<Vec<usize>>>)> = parts
        .into_iter()
        .filter_map(|(p, fc)| match &fc {
            Some(f) if ConvexHull::of_part(&p, Some(f)).is_err() => ConvexHull::new(&p).is_ok().then_some((p, None)),
            _ => Some((p, fc)),
        })
        .collect();
    if parts.is_empty() {
        return Err("no hull spans a solid".into());
    }
    let mass: Vec<_> = parts
        .iter()
        .enumerate()
        .map(|(k, (p, fc))| ConvexHull::of_part(p, fc.as_deref()).map(|h| h.unit_mass()).map_err(|e| format!("hull {k}: {e}")))
        .collect::<Result<_, _>>()?;
    // Flat (zero-volume) hulls baked into a compound add no volume and no mass, and their
    // centroid is undefined: leave them out while the fragment keeps real hulls (rubble's
    // importer does the same). A fragment of flat hulls only is rejected (equivalent box).
    let largest = mass.iter().map(|m| m.volume).fold(0.0f64, f64::max);
    let keep: Vec<bool> = mass.iter().map(|m| m.volume > 1e-9 * largest).collect();
    if largest <= 0.0 {
        return Err("every hull has zero volume".into());
    }
    let parts: Vec<_> = parts.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| p).collect();
    let mass: Vec<_> = mass.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(m, _)| m).collect();
    let volume: f64 = mass.iter().map(|m| m.volume).sum();
    let center = mass.iter().fold(Vec3::ZERO, |c, m| c + m.centroid * m.volume) / volume;
    let mut inertia = Mat3::ZERO;
    for m in &mass {
        let r = m.centroid - center;
        inertia += m.inertia + (Mat3::IDENTITY * r.dot(r) - Mat3::outer(r, r)) * m.volume;
    }
    let (_, mut v) = sym_eigen(inertia.m);
    let det = v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) - v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0]) + v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]);
    if det < 0.0 {
        for row in v.iter_mut() {
            row[2] = -row[2];
        }
    }
    // Columns of `v` are the principal axes: chunk-frame coordinates are `v^T (p - c)`.
    let local = |p: Vec3| {
        let d = p - center;
        Vec3::new(v[0][0] * d.x + v[1][0] * d.y + v[2][0] * d.z, v[0][1] * d.x + v[1][1] * d.y + v[2][1] * d.z, v[0][2] * d.x + v[1][2] * d.y + v[2][2] * d.z)
    };
    let compound: Vec<PartDesc> = parts.iter().map(|(p, fc)| PartDesc { points: p.iter().map(|q| local(*q).to_array()).collect(), faces: fc.clone() }).collect();
    let half = compound.iter().flat_map(|p| p.points.iter()).fold(Vec3::ZERO, |h, p| h.component_max(Vec3::from_array(*p).abs()));
    let mut c = chunk(center, half, mat);
    c.orientation = quat_wxyz(v);
    c.compound = Some(compound);
    Ok(c)
}

/// Load an exported Shardwright asset as a scene with one body.
pub fn import(path: &std::path::Path, name: &str) -> Result<Scene, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let json: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    import_value(&json, name).map_err(|e| format!("{}: {e}", path.display()))
}

/// [`import`] from an already parsed `shardwright-stress/1` document (e.g. one written
/// in memory from a fracture GLB by rubble's `engine-fracture`).
pub fn import_value(json: &Value, name: &str) -> Result<Scene, String> {
    if json["format"] != "shardwright-stress/1" {
        return Err("not a shardwright-stress/1 export".into());
    }
    let level = json["bond_level"].as_u64().ok_or("no bond_level")?;
    let frags = json["fragments"].as_array().ok_or("no fragments")?;
    let by_id: std::collections::HashMap<u64, &Value> = frags.iter().map(|f| (f["id"].as_u64().unwrap(), f)).collect();
    let names: std::collections::HashMap<u64, &str> = json["component_names"]
        .as_array()
        .map(|a| a.iter().filter_map(|n| Some((n[0].as_u64()?, n[1].as_str()?))).collect())
        .unwrap_or_default();
    fn ancestor<'a>(by_id: &std::collections::HashMap<u64, &'a Value>, mut f: &'a Value, at: u64) -> Option<u64> {
        while f["level"].as_u64()? > at {
            f = by_id.get(&(f["parent"].as_i64()? as u64))?;
        }
        f["id"].as_u64()
    }
    let mut scene = new_scene(name, &format!("Shardwright asset '{}', level {level}", json["name"].as_str().unwrap_or(name)));
    let mut chunks = Vec::new();
    let mut boxed = Vec::new();
    let mut local = std::collections::HashMap::new();
    for f in frags.iter().filter(|f| f["level"].as_u64() == Some(level)) {
        let (mat, _) = map_material(f["material"].as_str().unwrap_or(""));
        let mut c = match compound_chunk(f, mat) {
            Ok(c) => c,
            Err(e) => {
                if f["hulls"].is_array() {
                    boxed.push(format!("{}: {e}", f["id"]));
                }
                equivalent_box(f, mat).map_err(|e| format!("fragment {}: {e}", f["id"]))?
            }
        };
        let role = ROLES.get(f["role"].as_u64().unwrap_or(0) as usize).copied().unwrap_or("generic");
        c.groups.push(role.to_string());
        if let Some(m) = ancestor(&by_id, f, 0) {
            c.groups.push(format!("member:{m}"));
        }
        if let Some(n) = f["component"].as_u64().and_then(|k| names.get(&k)) {
            c.groups.push(format!("name:{n}"));
        }
        if level > 1 {
            if let Some(p) = ancestor(&by_id, f, 1) {
                c.groups.push(format!("l1:{p}"));
            }
        }
        scene.materials.entry(mat.to_string()).or_insert_with(|| material_preset(mat));
        local.insert(f["id"].as_u64().unwrap(), chunks.len());
        chunks.push(c);
    }
    let mut bonds = Vec::new();
    let mut anchors = 0usize;
    for b in json["bonds"].as_array().ok_or("no bonds")? {
        let a = local[&b["a"].as_u64().ok_or("bond a")?];
        let area = b["area"].as_f64().unwrap_or(0.0);
        if b["b"].as_i64() == Some(-1) {
            chunks[a].support = Support::Fixed;
            anchors += 1;
            continue;
        }
        let bb = local[&(b["b"].as_i64().ok_or("bond b")? as u64)];
        let mut normal = zup(&b["normal"]).normalized();
        let (ca, cb) = (Vec3::from_array(chunks[a].center), Vec3::from_array(chunks[bb].center));
        if normal.dot(cb - ca) < 0.0 {
            normal = -normal;
        }
        let eh = [b["extent_half"][0].as_f64().unwrap_or(0.0), b["extent_half"][1].as_f64().unwrap_or(0.0)];
        if !(area > 0.0 && eh[0] > 0.0 && eh[1] > 0.0) {
            continue;
        }
        // The oriented extent scaled to the bond's area (aspect kept).
        let s = (area / (4.0 * eh[0] * eh[1])).sqrt();
        let width = [2.0 * eh[0] * s, 2.0 * eh[1] * s];
        let mortar = b["composition"].as_array().is_some_and(|c| c.iter().any(|k| k[0].as_u64() == Some(1)));
        let chunk_mat = chunks[a].material.clone();
        let joint = if mortar { "mortar" } else { map_material(by_id[&b["a"].as_u64().unwrap()]["material"].as_str().unwrap_or("")).1 };
        scene.materials.entry(joint.to_string()).or_insert_with(|| material_preset(joint));
        let rebar = (chunk_mat == "concrete" && joint == "concrete").then(|| {
            scene.materials.entry("steel".into()).or_insert_with(Material::steel);
            let asset = b["rebar_area"].as_f64().unwrap_or(0.0);
            let area = if asset > 0.0 { asset } else { area * minimum_reinforcement(&Material::concrete(), &Material::steel()) };
            RebarSpec { area, material: "steel".into() }
        });
        bonds.push(BondDesc {
            a,
            b: bb,
            centroid: zup(&b["centroid"]).to_array(),
            normal: normal.to_array(),
            tangent: zup(&b["frame_u"]).to_array(),
            area,
            width,
            material: joint.to_string(),
            rebar,
            buckling_length: None,
            level: 0,
            groups: Vec::new(),
            derived: None,
        });
    }
    let compound = chunks.iter().filter(|c| c.compound.is_some()).count();
    if !boxed.is_empty() {
        eprintln!("shardwright: {} fragments as equivalent-inertia boxes (hulls rejected): {}", boxed.len(), boxed.join("; "));
    }
    let shapes = if compound == chunks.len() { "compound chunks of the asset's hulls".to_string() } else { format!("{compound} compound chunks, {} equivalent-inertia boxes", chunks.len() - compound) };
    scene.description = format!("{} ({} chunks, {} bonds, {anchors} anchors; {shapes})", scene.description, chunks.len(), bonds.len());
    // Compound chunks take the penalty contact.
    if compound > 0 {
        scene.sim.methods.layer_contact = false;
    }
    scene.materials.entry("concrete".into()).or_insert_with(Material::concrete);
    scene.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
    scene.bodies.push(body(name, chunks, bonds));
    scene.validate()?;
    Ok(scene)
}

fn bounds(scene: &Scene) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
    for c in scene.bodies.iter().flat_map(|b| b.chunks.iter()) {
        let (p, h) = (Vec3::from_array(c.center), Vec3::from_array(c.half_extents));
        lo = lo.component_min(p - h);
        hi = hi.component_max(p + h);
    }
    (lo, hi)
}

/// One of the destruction cases on an imported asset (`import`): `idle` (the intact
/// asset at rest), `ram` (stress-gpu's `scenes::ram_at`: a 400 kg box at 12 m/s into
/// the -x face, mid-extent), `projectile` (frac-sim's default steel ball) or `remove`
/// (the sudden removal of the authored component `remove`, e.g. `f0_col_21`).
pub fn destruction_case(base: &Scene, case: &str, remove: &str) -> Result<Scene, String> {
    let mut s = base.clone();
    s.name = format!("{}_{case}", base.name);
    match case {
        "idle" => {}
        // stress-gpu `scenes::ram_at`: a 400 kg box at 12 m/s into the -x face, at the
        // middle of the asset's extent.
        "ram" => {
            let (lo, hi) = bounds(&s);
            let mid = (lo + hi) * 0.5;
            let material = s.bodies[0].chunks[0].material.clone();
            s.impactors.push(ImpactorDesc {
                name: "ram".into(),
                shape: ImpactorShape::Box { half_extents: [0.15, 0.3, 0.3] },
                mass: 400.0,
                position: [lo.x - 0.35, mid.y, mid.z],
                orientation: [1.0, 0.0, 0.0, 0.0],
                velocity: [12.0, 0.0, 0.0],
                angular_velocity: [0.0; 3],
                material,
                crush: None,
            });
        }
        // frac-sim's defaults (tools/sim/src/main.rs): a steel ball, r 0.4 m, density
        // 7800, at (13, 1.4, 8.15) moving (-15, 0, 0) in the asset's +Y-up frame; the
        // importer's +Z-up frame is (x, -z, y).
        "projectile" => {
            let r = 0.4;
            s.materials.entry("steel".into()).or_insert_with(Material::steel);
            s.impactors.push(ImpactorDesc {
                name: "ball".into(),
                shape: ImpactorShape::Sphere { radius: r },
                mass: 7800.0 * 4.0 / 3.0 * std::f64::consts::PI * r * r * r,
                position: [13.0, -8.15, 1.4],
                orientation: [1.0, 0.0, 0.0, 0.0],
                velocity: [-15.0, 0.0, 0.0],
                angular_velocity: [0.0; 3],
                material: "steel".into(),
                crush: None,
            });
        }
        // Sudden removal (GSA alternate path) of one authored component, selected by the
        // `name:` group the importer gives each chunk.
        "remove" => {
            let group = format!("name:{remove}");
            let n = s.bodies[0].chunks.iter().filter(|c| c.groups.contains(&group)).count();
            if n == 0 {
                return Err(format!("no chunks in component {remove} (export without the bake's asset.json?)"));
            }
            eprintln!("removing {remove}: {n} chunks");
            s.events.push(EventDesc::RemoveChunks { time: 0.1, duration: 0.0, body: s.bodies[0].name.clone(), chunks: ChunkSelector::Group(group) });
        }
        other => return Err(format!("unknown case {other}")),
    }
    Ok(s)
}

/// Runs `world` over its scene's duration, timing every complete frame, and prints
/// progress every `every` frames and a summary line. The observation carries the
/// timings: `setup_s` (given), `first_frame_ms` (gravity prestress, first
/// factorization), and the peak and mean of the others.
pub fn run_timed(scene: &Scene, mut world: World, setup: f64, every: u64) -> Observation {
    use web_time::Instant;
    let (chunks, bonds): (usize, usize) = scene.bodies.iter().fold((0, 0), |(c, b), body| (c + body.chunks.len(), b + body.bonds.len()));
    eprintln!("{}: {chunks} chunks, {bonds} bonds, mode {:?}, {} s, engine {}", scene.name, scene.sim.solve_mode, scene.sim.duration, world.engine_name().unwrap_or("standalone"));
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as u64;
    let (mut first, mut peak, mut peak_frame, mut total) = (0.0f64, 0.0f64, 0u64, 0.0f64);
    while world.frame < frames {
        let substeps = world.solver.substeps;
        let t = Instant::now();
        world.step_frame();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        total += ms;
        if world.frame == 1 {
            first = ms;
        } else if ms > peak {
            (peak, peak_frame) = (ms, world.frame);
        }
        if world.frame % every == 0 || world.frame == frames {
            eprintln!(
                "  t={:.3} frame {:>4} {:>8.1} ms {:>6} substeps  broken {:>5}  clusters {:>4}  max_u {:.2}",
                world.solver.time,
                world.frame,
                ms,
                world.solver.substeps - substeps,
                world.solver.broken_bond_count(),
                world.solver.clusters.len(),
                world.solver.max_utilization
            );
        }
    }
    let mut obs = world.observation();
    obs.values.insert("setup_s".into(), setup);
    obs.values.insert("first_frame_ms".into(), first);
    obs.values.insert("peak_frame_ms".into(), peak);
    obs.values.insert("peak_frame".into(), peak_frame as f64);
    obs.values.insert("mean_frame_ms".into(), total / frames.max(1) as f64);
    let v = |k: &str| obs.values.get(k).copied().unwrap_or(f64::NAN);
    println!(
        "{:<36} chunks {chunks} bonds {bonds}  broken {:>5}  fragments {:>4}  collapse {}  max_u {:.2}  bond_dissipation {:.3e} J  setup {:.1} s  first {:.0} ms  peak {:.0} ms (frame {peak_frame})  mean {:.0} ms",
        scene.name,
        v("broken_bonds"),
        v("fragments"),
        obs.flags.get("collapse").copied().unwrap_or(false),
        v("max_failure_index"),
        v("bond_dissipation"),
        setup,
        first,
        peak,
        v("mean_frame_ms"),
    );
    obs
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// A case from a command line `<export.json> <case> [--remove NAME] [--duration S]
/// [--every N] [--out DIR] [--feature NAME=on|off]... [--set path=json]...`: its scene
/// (written to `DIR/<case>.scene.json`), the case, the output directory and the progress
/// interval in frames.
pub fn case_from_args(args: &[String]) -> Result<(Scene, String, Option<std::path::PathBuf>, u64), String> {
    let (path, case) = match (args.first(), args.get(1)) {
        (Some(p), Some(c)) => (std::path::Path::new(p), c.clone()),
        _ => return Err("usage: <export.json> <idle|ram|projectile|remove> [opts]".into()),
    };
    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
    let base = crate::scene_pack::import(path, &stem, "structure")?;
    let mut scene = destruction_case(&base, &case, &flag(args, "--remove").unwrap_or_else(|| "f0_col_21".into()))?;
    for (i, a) in args.iter().enumerate() {
        match a.as_str() {
            "--feature" => {
                let (n, v) = args.get(i + 1).and_then(|f| f.split_once('=')).ok_or("--feature NAME=on|off")?;
                scene = scene.with_feature(n, v == "on")?;
            }
            "--set" => {
                let (p, v) = args.get(i + 1).and_then(|f| f.split_once('=')).ok_or("--set path=json")?;
                scene = scene.with_override(p, v)?;
            }
            _ => {}
        }
    }
    if let Some(d) = flag(args, "--duration") {
        scene.sim.duration = d.parse().map_err(|e| format!("--duration: {e}"))?;
    }
    let every: u64 = flag(args, "--every").map(|v| v.parse().unwrap()).unwrap_or(6);
    let out = flag(args, "--out").map(std::path::PathBuf::from);
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(format!("{case}.scene.json")), serde_json::to_string(&scene).unwrap()).map_err(|e| e.to_string())?;
    }
    Ok((scene, case, out, every))
}
