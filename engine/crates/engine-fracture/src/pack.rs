//! Scene packs (the blast-stress-solver / vibe-land / vibe-city JSON: `scenario.nodes`,
//! `scenario.bonds`, per-node box or convex-hull colliders, optional `nodeMeshes`) as lab
//! scenes. The vibe-city `/destructible-stress` presets are generated into this format by
//! `engine/scenes/vibe-city` (`bun run export.ts`).
//!
//! - Nodes with mass become dynamic bodies; mass-0 nodes (supports) become static bodies,
//!   and bonds to them weld to the world.
//! - Box nodes keep SHAPE_BOX colliders; convex-hull nodes are cooked from their points
//!   (`ConvexHull::from_faces` when the pack lists faces, else `from_points`; a `compound`
//!   node has several exact convex parts) and
//!   placed in their principal inertia frame. A hull over
//!   the engine's vertex limit is an error.
//! - Bonds become the same breakable welds as baked bonds (`Joint`, `scale_bond_springs`).
//!
//! Bond capacities come from the pack's Blast limits. blast-stress-solver measures bond
//! stress as impulse / area (`computeBondStress`), with impulses accumulated over one
//! solver step of `BLAST_DT`; a limit L therefore holds a force of L * area / BLAST_DT.
//! Bending uses the same law: the angular term reads M * 2 / len per area, so the moment
//! capacity is L_tension * area * len / 2 / BLAST_DT (len = node centroid distance).
//! Blast has no bond stiffness (it holds actors rigid); springs use one modulus
//! (`PACK_YOUNGS`) along each node's distance to the bond, square-face second moments, and
//! the lab's scene-wide penalty scale, so only their ratios matter.

use crate::bonds::{scale_bond_springs, BOND_TICK};
use crate::math::*;
use crate::scene::*;
use anyhow::{anyhow, bail, Context, Result};
use engine_schema::convex::ConvexHull;
use glam::Vec3;
use serde_json::Value;

/// The step blast-stress-solver's impulses are accumulated over (vibe-city runs 60 Hz).
pub const BLAST_DT: f64 = 1.0 / 60.0;
/// Bond spring modulus (concrete), Pa. Shear modulus is E / 2.4.
pub const PACK_YOUNGS: f64 = 30e9;

const SUPPORT_COLOR: [f64; 3] = [0.42, 0.43, 0.46];

/// The pack's own setup, for the host's shot and status lines.
#[derive(Clone, Debug, Default)]
pub struct PackInfo {
    pub key: String,
    pub title: String,
    pub nodes: usize,
    pub supports: usize,
    pub bonds: usize,
    /// Each node's frame in the scene (the footprint shift applied; axes the scene's): where
    /// its collider's points are. Its body's pose is the frame for a box, but a hull's body sits
    /// at its centre of mass on its principal axes.
    pub node_frames: Vec<V3>,
}

fn v3(v: &Value) -> Result<V3> {
    let c = |k: &str| v[k].as_f64().ok_or_else(|| anyhow!("missing {k}"));
    Ok([c("x")?, c("y")?, c("z")?])
}

fn f(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

/// A unit vector perpendicular to `n`.
fn perpendicular(n: V3) -> V3 {
    let a = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    norm(cross(n, a))
}

/// Small per-node shade change so neighbouring chunks read apart.
fn shade(base: [f64; 3], i: usize) -> [f64; 3] {
    let h = (i as u32).wrapping_mul(2654435761) >> 24;
    let k = 0.9 + 0.2 * (h as f64 / 255.0);
    base.map(|c| (c * k).min(1.0))
}

/// A `#rrggbb` colour as linear-ish [0, 1] channels (as the key colours are given).
fn hex_color(v: &Value) -> Option<[f64; 3]> {
    let h = v.as_str()?.strip_prefix('#')?;
    if h.len() != 6 {
        return None;
    }
    let c = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok().map(|x| x as f64 / 255.0);
    Some([c(0)?, c(2)?, c(4)?])
}

fn base_color(key: &str) -> [f64; 3] {
    let k = key.to_ascii_lowercase();
    if k.contains("brick") {
        [0.66, 0.33, 0.24]
    } else if k.contains("bridge") {
        [0.62, 0.62, 0.6]
    } else if k.contains("fractured") || k.contains("glb") {
        [0.74, 0.7, 0.62]
    } else {
        [0.72, 0.69, 0.64]
    }
}

struct NodeBody {
    /// Body index (static for supports).
    body: u32,
    q: [f64; 4],
    support: bool,
}

/// Build a lab scene from scene-pack JSON bytes.
pub fn build_pack_scene(bytes: &[u8]) -> Result<(Scene, PackInfo)> {
    let json: Value = serde_json::from_slice(bytes).context("scene pack is not JSON")?;
    let sc = &json["scenario"];
    let nodes = sc["nodes"]
        .as_array()
        .ok_or_else(|| anyhow!("pack has no scenario.nodes"))?;
    let bonds = sc["bonds"]
        .as_array()
        .ok_or_else(|| anyhow!("pack has no scenario.bonds"))?;
    let sizes = sc["nodeSizes"].as_array();
    let colliders = sc["nodeColliders"].as_array();
    let library = sc["shapeLibrary"].as_array();
    let meshes = json["nodeMeshes"].as_array();
    let key = json["key"].as_str().unwrap_or("pack").to_string();
    let limits = &json["defaults"]["solver"]["limits"];
    let limit = |k: &str| f(limits, k).unwrap_or(f64::INFINITY);
    let (l_tension, l_compression, l_shear) = (
        limit("tensionFatal"),
        limit("compressionFatal"),
        limit("shearFatal"),
    );
    // ScenePack v2 gives the limits per material (`defaults.solver.materials[m]`, a bond's
    // `m`); a pack-wide `limits` block, when present, still applies to every bond.
    let materials = json["defaults"]["solver"]["materials"].as_array();
    let bond_limits = |b: &Value| -> (f64, f64, f64) {
        if !limits.is_null() {
            return (l_tension, l_compression, l_shear);
        }
        let m = b["m"].as_u64().and_then(|m| materials.and_then(|ms| ms.get(m as usize)));
        match m {
            Some(m) => (
                f(m, "tensionFatal").unwrap_or(f64::INFINITY),
                f(m, "compressionFatal").unwrap_or(f64::INFINITY),
                f(m, "shearFatal").unwrap_or(f64::INFINITY),
            ),
            None => (l_tension, l_compression, l_shear),
        }
    };
    let color = base_color(&key);

    let centroid: Vec<V3> = nodes
        .iter()
        .map(|n| v3(&n["centroid"]))
        .collect::<Result<_>>()?;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    let half_of = |i: usize| -> V3 {
        let s = sizes
            .and_then(|s| s.get(i))
            .and_then(|s| v3(s).ok())
            .unwrap_or([0.5, 0.5, 0.32]);
        s.map(|x| (x * 0.5).max(0.05))
    };
    for (i, c) in centroid.iter().enumerate() {
        let h = half_of(i);
        for k in 0..3 {
            lo[k] = lo[k].min(c[k] - h[k]);
            hi[k] = hi[k].max(c[k] + h[k]);
        }
    }
    if !lo.iter().all(|x| x.is_finite()) {
        bail!("scene pack has no nodes");
    }
    // Centre the footprint on x = z = 0 (layouts assume it); lift anything below the floor.
    let shift = [
        -(lo[0] + hi[0]) / 2.0,
        (-lo[1]).max(0.0),
        -(lo[2] + hi[2]) / 2.0,
    ];

    let mut bodies = vec![Body::fixed([0.0; 3])];
    let mut cols = vec![Collider::ground()];
    let mut arena = hull_arena(None)?;
    let mut vertices = Vec::new();
    let mut map: Vec<NodeBody> = Vec::with_capacity(nodes.len());
    let mut info = PackInfo {
        key: key.clone(),
        title: json["title"].as_str().unwrap_or(&key).to_string(),
        nodes: nodes.len(),
        bonds: bonds.len(),
        ..Default::default()
    };

    for (i, n) in nodes.iter().enumerate() {
        let mass = f(n, "mass").unwrap_or(0.0);
        let support = mass.is_nan() || mass <= 0.0;
        let half = half_of(i);
        let at = add(centroid[i], shift);
        info.node_frames.push(at);
        let id = bodies.len() as u32;
        // A node's material colour where the pack gives one, else the pack's key colour.
        let own = nodes[i]["m"]
            .as_u64()
            .and_then(|m| materials.and_then(|ms| ms.get(m as usize)))
            .and_then(|m| hex_color(&m["color"]));
        let tint = if support {
            SUPPORT_COLOR
        } else {
            shade(own.unwrap_or(color), i)
        };
        let collider = colliders.and_then(|c| c.get(i));
        let kind = collider.and_then(|c| c["kind"].as_str()).unwrap_or("");
        let point_list = |v: &Value| -> Vec<Vec3> {
            v.as_array().map_or(vec![], |p| {
                p.chunks_exact(3)
                    .map(|t| {
                        Vec3::new(
                            t[0].as_f64().unwrap_or(0.0) as f32,
                            t[1].as_f64().unwrap_or(0.0) as f32,
                            t[2].as_f64().unwrap_or(0.0) as f32,
                        )
                    })
                    .collect()
            })
        };
        // Convex parts in the node frame: one hull, or the exact parts of a compound. A hull
        // over the engine's vertex limit is an error (re-export the pack), never simplified.
        // Each part: points, and its faces when the pack lists them (cooked like baked
        // hulls, `from_faces`); without faces the cooker builds the hull from the points.
        let part_of = |p: &Value| -> (Vec<Vec3>, Option<(Vec<u32>, Vec<u32>)>) {
            let faces = p["faces"].as_array().map(|faces| {
                let lists: Vec<Vec<u32>> = faces
                    .iter()
                    .map(|f| {
                        f.as_array().map_or(vec![], |f| {
                            f.iter()
                                .filter_map(|x| x.as_u64().map(|x| x as u32))
                                .collect()
                        })
                    })
                    .collect();
                (
                    lists.iter().map(|f| f.len() as u32).collect(),
                    lists.into_iter().flatten().collect(),
                )
            });
            (point_list(&p["points"]), faces)
        };
        // A `shape` collider names an entry of `scenario.shapeLibrary` (shared by many nodes,
        // in each node's frame): that entry's hull or compound. An unknown kind is an error, never
        // the node's box.
        let parts_of = |c: &Value| -> Vec<(Vec<Vec3>, Option<(Vec<u32>, Vec<u32>)>)> {
            match c["kind"].as_str().unwrap_or("") {
                "compound" => c["parts"]
                    .as_array()
                    .map_or(vec![], |parts| parts.iter().map(part_of).collect()),
                _ => vec![part_of(c)],
            }
        };
        let part_data: Vec<(Vec<Vec3>, Option<(Vec<u32>, Vec<u32>)>)> = match (kind, collider) {
            ("convex_hull" | "compound", Some(c)) => parts_of(c),
            ("shape", Some(c)) => {
                let k = c["shape"]
                    .as_u64()
                    .ok_or_else(|| anyhow!("node {i}: shape collider without a shape index"))?;
                let entry = library
                    .and_then(|l| l.get(k as usize))
                    .ok_or_else(|| anyhow!("node {i}: shape {k} is not in scenario.shapeLibrary"))?;
                match entry["kind"].as_str().unwrap_or("convex_hull") {
                    "convex_hull" | "compound" => parts_of(entry),
                    other => bail!("node {i}: shapeLibrary[{k}] has kind '{other}'"),
                }
            }
            ("cuboid", _) | ("", None) => vec![],
            (other, _) => bail!("node {i}: collider kind '{other}' is not supported"),
        };
        let parts: Vec<ConvexHull> = part_data
            .iter()
            .map(|(p, faces)| {
                match faces {
                    Some((sizes, indices)) => ConvexHull::from_faces(p, sizes, indices),
                    None => ConvexHull::from_points(p),
                }
                .map_err(|e| anyhow!("node {i}: {e}"))
            })
            .collect::<Result<_>>()?;
        let box_half = collider
            .filter(|_| kind == "cuboid")
            .and_then(|c| c["halfExtents"].as_array())
            .filter(|h| h.len() == 3)
            .map(|h| [0, 1, 2].map(|k| h[k].as_f64().unwrap_or(half[k])))
            .unwrap_or(half);

        // Unit-density volume, centre and inertia of the union of the parts.
        let union = (!parts.is_empty()).then(|| {
            let props: Vec<_> = parts.iter().map(|h| h.mass_properties(1.0)).collect();
            let volume: f64 = props.iter().map(|m| m.volume as f64).sum();
            let com = props.iter().fold([0.0; 3], |c, m| {
                add(
                    c,
                    scale(f64x3(m.center_of_mass.to_array()), m.volume as f64),
                )
            });
            let com = scale(com, 1.0 / volume.max(1e-12));
            let mut inertia = [0.0f64; 9];
            for m in &props {
                let cols = m.inertia.to_cols_array();
                let d = sub(f64x3(m.center_of_mass.to_array()), com);
                let (v, dd) = (m.volume as f64, dot(d, d));
                for r in 0..3 {
                    for c in 0..3 {
                        let shift = v * (if r == c { dd } else { 0.0 } - d[r] * d[c]);
                        inertia[r * 3 + c] += cols[c * 3 + r] as f64 + shift;
                    }
                }
            }
            (volume, com, inertia)
        });

        // Pose and mass: hulls in their principal frame, boxes axis-aligned.
        let (p, q, inv_i, local) = match union {
            Some((volume, com, inertia)) if !support => {
                let rows = inertia.map(|x| x as f32);
                let (q, moments) = principal_axes(&rows)?;
                let scale_i = mass / volume.max(1e-12);
                (add(at, com), q, moments.map(|m| 1.0 / (m * scale_i)), com)
            }
            Some((_, com, _)) => (add(at, com), IDENTITY, [0.0; 3], com),
            None => {
                let [hx, hy, hz] = box_half;
                let inv_i = if support {
                    [0.0; 3]
                } else {
                    [
                        3.0 / (mass * (hy * hy + hz * hz)),
                        3.0 / (mass * (hx * hx + hz * hz)),
                        3.0 / (mass * (hx * hx + hy * hy)),
                    ]
                };
                (at, IDENTITY, inv_i, [0.0; 3])
            }
        };
        bodies.push(if support {
            Body {
                q,
                ..Body::fixed(p)
            }
        } else {
            Body {
                p,
                q,
                v: [0.0; 3],
                w: [0.0; 3],
                inv_mass: 1.0 / mass,
                inv_i,
                flags: FLAG_DYNAMIC | FLAG_SLEEPING,
            }
        });
        if support {
            info.supports += 1;
        }

        match union {
            Some((volume, _, _)) => {
                let to_body = |v: V3| inverse_rotate(q, sub(v, local));
                let density = if support {
                    0.0
                } else {
                    mass / volume.max(1e-12)
                };
                // One collider per part, each cooked once in the body frame (re-validating
                // rotated faces fails at f32).
                let mut cooked = Vec::with_capacity(parts.len());
                for (h, (points, faces)) in parts.iter().zip(&part_data) {
                    let body_hull = match faces {
                        // The listed faces, rotated: the check allows the node frame's f32 rounding.
                        Some((sizes, indices)) => {
                            let scale = points
                                .iter()
                                .fold(0f32, |m, p| m.max(p.abs().max_element()));
                            let body_points: Vec<Vec3> = points
                                .iter()
                                .map(|v| Vec3::from(f32x3(to_body(f64x3(v.to_array())))))
                                .collect();
                            // Same corners either way: if the rotation's f32 rounding fails the
                            // face check, let the cooker rebuild the faces from the corners.
                            ConvexHull::from_faces_in_frame(&body_points, sizes, indices, scale)
                                .or_else(|_| ConvexHull::from_points(&body_points))
                        }
                        None => {
                            let body_points: Vec<Vec3> = h
                                .vertices
                                .iter()
                                .map(|v| Vec3::from(f32x3(to_body(f64x3(v.to_array())))))
                                .collect();
                            ConvexHull::from_points(&body_points)
                        }
                    };
                    let (body_hull, offset) = body_hull
                        .map_err(|e| anyhow!("node {i}: {e}"))?
                        .recentered();
                    let base = arena
                        .insert(&body_hull)
                        .map_err(|e| anyhow!("node {i}: {e}"))?;
                    let offset = f64x3(offset.to_array());
                    cols.push(Collider::hull(
                        id,
                        density,
                        base,
                        1.0,
                        body_hull.bounding_radius(),
                        offset,
                    ));
                    cooked.push((body_hull, offset));
                }
                // Render: the node mesh if present, else the parts' faces.
                let mesh = meshes.and_then(|m| m.get(i));
                let mut drawn = false;
                if let Some(m) = mesh {
                    let pos: Vec<f64> = m["positions"]
                        .as_array()
                        .map_or(vec![], |a| a.iter().filter_map(Value::as_f64).collect());
                    let idx: Vec<usize> = m["indices"].as_array().map_or(vec![], |a| {
                        a.iter()
                            .filter_map(|x| x.as_u64().map(|x| x as usize))
                            .collect()
                    });
                    let count = pos.len() / 3;
                    let tris: Vec<usize> = if idx.is_empty() {
                        (0..count).collect()
                    } else {
                        idx
                    };
                    let at_v = |k: usize| [pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]];
                    for t in tris
                        .chunks_exact(3)
                        .filter(|t| t.iter().all(|&k| k < count))
                    {
                        let pts = [0, 1, 2].map(|k| to_body(at_v(t[k])));
                        let nrm = norm(cross(sub(pts[1], pts[0]), sub(pts[2], pts[0])));
                        for p in pts {
                            vertex(&mut vertices, p, nrm, tint, id);
                        }
                        drawn = true;
                    }
                }
                if !drawn {
                    for (body_hull, offset) in &cooked {
                        for face in &body_hull.faces {
                            let nrm = f64x3(face.normal.to_array());
                            let ids: Vec<usize> =
                                face.indices.iter().map(|&k| k as usize).collect();
                            for k in 1..ids.len() - 1 {
                                for &j in &[ids[0], ids[k], ids[k + 1]] {
                                    let v = add(f64x3(body_hull.vertices[j].to_array()), *offset);
                                    vertex(&mut vertices, v, nrm, tint, id);
                                }
                            }
                        }
                    }
                }
            }
            None => {
                let [hx, hy, hz] = box_half;
                let density = if support {
                    0.0
                } else {
                    mass / (8.0 * hx * hy * hz)
                };
                cols.push(Collider::new(id, SHAPE_BOX, density, [hx, hy, hz, 0.0]));
                for (nrm, quad) in crate::cuboid::FACES {
                    for k in [0, 1, 2, 0, 2, 3] {
                        let s = quad[k];
                        vertex(
                            &mut vertices,
                            [s[0] * hx, s[1] * hy, s[2] * hz],
                            nrm,
                            tint,
                            id,
                        );
                    }
                }
            }
        }
        map.push(NodeBody {
            body: id,
            q,
            support,
        });
    }

    // Bonds: breakable welds; a bond to a support welds the dynamic side to the world.
    let mut joints = Vec::new();
    let mut lines = Vec::new();
    let (e, g) = (PACK_YOUNGS, PACK_YOUNGS / 2.4);
    for b in bonds {
        let (Some(i0), Some(i1)) = (b["node0"].as_u64(), b["node1"].as_u64()) else {
            continue;
        };
        let (i0, i1) = (i0 as usize, i1 as usize);
        if i0 >= map.len() || i1 >= map.len() || i0 == i1 {
            continue;
        }
        let mut n = norm(v3(&b["normal"])?);
        let c = add(v3(&b["centroid"])?, shift);
        let area = f(b, "area").unwrap_or(0.0).max(1e-8);
        let (l_tension, l_compression, l_shear) = bond_limits(b);
        let (mut a, mut o) = (i0, i1);
        if map[a].support {
            if map[o].support {
                continue;
            }
            std::mem::swap(&mut a, &mut o);
            n = scale(n, -1.0);
        }
        let ba = &map[a];
        let bo = &map[o];
        let world = bo.support;
        let pa = bodies[ba.body as usize].p;
        let po = bodies[bo.body as usize].p;
        let len = length(sub(centroid[i1], centroid[i0])).max(1e-6);
        let da = dot(sub(c, pa), n).abs().max(1e-6);
        let db = if world {
            0.0
        } else {
            dot(sub(po, c), n).abs().max(1e-6)
        };
        let ce = (da + db) / e;
        let cg = (da + db) / g;
        let i_face = area * area / 12.0;
        let u = perpendicular(n);
        joints.push(Joint {
            body_a: ba.body,
            body_b: if world { 0 } else { bo.body },
            anchor_a: inverse_rotate(ba.q, sub(c, pa)),
            anchor_b: if world {
                c
            } else {
                inverse_rotate(bo.q, sub(c, po))
            },
            rest: if world {
                conjugate(ba.q)
            } else {
                quat_mul(conjugate(ba.q), bo.q)
            },
            normal: inverse_rotate(ba.q, n),
            frame_u: inverse_rotate(ba.q, u),
            strength: [
                (l_tension * area / BLAST_DT).max(1e-6),
                (l_compression * area / BLAST_DT).max(1e-6),
                (l_shear * area / BLAST_DT).max(1e-6),
                (l_tension * area * len * 0.5 / BLAST_DT).max(1e-9),
            ],
            k: [
                area / ce,
                area / cg,
                area / cg,
                i_face / ce,
                i_face / ce,
                2.0 * i_face / cg,
            ],
            world,
        });
        let line = if world {
            [0.95, 0.6, 0.2]
        } else {
            [0.25, 0.9, 0.8]
        };
        // Line k is joint k (`fracture_draw.slang` hides it once the joint breaks). An
        // anchor's line ends at its point in the piece's frame, not on body 0: copies share
        // body 0, so a ground point would sit at the first copy's anchor.
        let n = |partner: u32| bond_line_normal(partner, pa, (!world).then_some(po));
        vertex(&mut lines, [0.0; 3], n(if world { 0 } else { bo.body }), line, ba.body);
        if world {
            vertex(&mut lines, inverse_rotate(ba.q, sub(c, pa)), n(0), line, ba.body);
        } else {
            vertex(&mut lines, [0.0; 3], n(ba.body), line, bo.body);
        }
    }
    let joints = scale_bond_springs(&joints, &bodies, BOND_TICK);

    let span = sub(hi, lo);
    let mut s = Scene::from_parts(
        bodies,
        cols,
        &arena,
        SceneParams::lab(LAB_FLAGS | FLAG_CONTACT_GAP),
    );
    let shot = &json["defaults"]["projectile"];
    let radius = f(shot, "radius").unwrap_or(0.5);
    add_projectiles(&mut s, radius, 2.0 * radius + 0.5);
    s.projectile_mass = f(shot, "mass").unwrap_or(1000.0);
    s.projectile_speed = f(shot, "speed");
    s.bond_count = joints.len() as u32;
    s.joints = joints;
    s.vertices = vertices;
    s.lines = lines;
    s.pieces = nodes.len() as u32;
    s.selected = (1..=nodes.len() as u32).collect();
    s.extent = span[0].max(span[1]).max(span[2]);
    s.extent_x = span[0];
    s.extent_z = span[2];
    s.height = span[1];
    s.center = [0.0, (lo[1] + shift[1]) + span[1] / 2.0, 0.0];
    s.placement = shift;
    s.grid_cell_size = grid_config(&s.colliders).0;
    Ok((s, info))
}
