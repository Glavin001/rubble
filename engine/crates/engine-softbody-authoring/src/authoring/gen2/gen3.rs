//! gen3: the render model ([`super::model`]) rigged with crash physics
//! ([`super::rig`]) and bound to it by construction ([`crate::flexskin`]):
//! every render vertex's node weights over its own part's lattice, written
//! with the export as a `.rskin` sidecar that the flexbody rows name.

use super::build::{Car, Part};
use super::export::Flex;
use super::rig::{self, Rig};
use crate::flexskin::{self, MeshSkin, Skin};
use crate::glb::Glb;
use glam::Vec3;

/// The skin sidecar's file name in the vehicle folder.
pub const SKIN_FILE: &str = "gen2.rskin";
/// The render model's file name in the vehicle folder.
pub const GLB_FILE: &str = "gen2.glb";

/// The most a glass vertex's skin may amplify its nodes' motion apart from each
/// other ([`flexskin::amplification`]: Σ|Aⱼ|, 1 where it interpolates them).
///
/// Glass spans an opening with no nodes in it. Where the frame's nodes on one side
/// are sparse (long kernels) and on the other dense (short ones), the pane's edge is
/// in reach of the one side alone and the affine fit extrapolates from it: the
/// generated 4x4's windscreen top bound −1.45 to the roof header's middle node
/// (Σ|A| 4.46), so a roof crush that pushed the header down lifted the glass up, a
/// grey spike 0.7–0.9 m above the car (`engine/docs/generated-vehicles-lessons.md`,
/// "Glass binding"). A glass mesh whose vertex passes this grows its kernels until
/// the frame on every side is in reach. **Choice**: 3.0, the most any other
/// design's glass reaches (the wagon's quarter lights, 2.98) and none of them
/// spikes in a crush; the 4x4's windscreen binds at 1.32 with it, its rear glass at
/// 1.45 (3.02), its quarter lights at 2.64 (3.43).
pub const GLASS_AMP_MAX: f64 = 3.0;

/// The physics part a model mesh rides (None: a wheel's piece).
pub fn part_of(mesh: &str) -> Option<Part> {
    let tail = |s: &str| mesh.ends_with(s);
    if mesh.starts_with("tyre_") || mesh.starts_with("rim_") || mesh.starts_with("brake_disc_") {
        return None;
    }
    if mesh.starts_with("door_")
        || mesh.starts_with("glass_door_")
        || mesh.starts_with("handle_")
        || mesh.starts_with("door_trim_")
    {
        let front = mesh.contains("_F");
        let left = tail("L");
        return Some(Part::door(front, left));
    }
    if let Some(side) = mesh.strip_prefix("glass_quarter_") {
        // The rear door's frame carries the quarter light.
        return Some(Part::door(false, side == "L"));
    }
    Some(match mesh {
        "mirror_L" => Part::DoorFL,
        "mirror_R" => Part::DoorFR,
        "hood" => Part::Hood,
        "trunk_lid" => Part::Trunk,
        "fender_FL" => Part::FenderL,
        "fender_FR" => Part::FenderR,
        "bumper_front" | "grille" => Part::BumperF,
        "bumper_rear" => Part::BumperR,
        "engine_block" => Part::Engine,
        _ => Part::Body,
    })
}

/// Pieces that move rigidly (one frame for the piece, [`flexskin::bind_rigid`]):
/// lamps, mirrors, handles, wipers, the grille, the cabin's hard parts, the
/// engine's ancillaries and the wheels' pieces.
pub fn rigid(mesh: &str) -> bool {
    [
        "mirror_",
        "handle_",
        "wiper_",
        "seat_",
        "steering_wheel",
        "dash",
        "engine_block",
        "radiator",
        "tyre_",
        "rim_",
        "brake_disc_",
    ]
    .iter()
    .any(|p| mesh.starts_with(p))
}

/// The nodes a mesh binds to: its part's structural nodes (no contact-only
/// lane nodes, latch hardware or wheel nodes), or a wheel's hub and rim nodes.
pub fn pool(car: &Car, mesh: &str) -> Vec<u32> {
    pool_with(car, mesh, &part_of)
}

/// [`pool`] with the meshes on parts by `part_of`.
pub fn pool_with(car: &Car, mesh: &str, part_of: &dyn Fn(&str) -> Option<Part>) -> Vec<u32> {
    // A mesh whose part's definition names its nodes binds to those.
    if let Some(b) = car.bind.get(mesh) {
        return b.nodes.iter().map(|&i| i as u32).collect();
    }
    match part_of(mesh) {
        Some(p) => (0..car.nodes.len() as u32)
            .filter(|&i| {
                let n = &car.nodes[i as usize];
                n.part == p
                    && !n.contact
                    && !n.no_group
                    && n.sub.is_none()
                    && !n.name.starts_with("hub")
            })
            .collect(),
        None => {
            let tag = &mesh[mesh.len() - 2..];
            let (w, x) = (&tag[..1], &tag[1..]);
            (0..car.nodes.len() as u32)
                .filter(|&i| {
                    let n = &car.nodes[i as usize].name;
                    *n == format!("hub{w}i{x}")
                        || *n == format!("hub{w}o{x}")
                        || (n.starts_with(&format!("rim{w}")) && n.ends_with(x))
                })
                .collect()
        }
    }
}

/// Per node of `car`: on its outer collision skin (a corner of a collision triangle
/// that is not an inner face). Vehicle contact holds these and only these: a car meets
/// another's live outer triangles (the paint guard's surface, the contact offsets', edge
/// contact's).
pub fn outer_skin_nodes(car: &Car) -> Vec<bool> {
    let mut on = vec![false; car.nodes.len()];
    for t in car.tris.iter().filter(|t| !t.inner) {
        for &i in &t.n {
            on[i] = true;
        }
    }
    on
}

/// A mesh drawn as the car's outer paint: a deforming mesh (not a rigid piece, not
/// glass) with a primitive in the `paint` material. It binds to nodes on the outer
/// collision skin alone ([`outer_skin_nodes`]): its own part's, and those of the outer
/// triangle under each of its vertices, which may be another part's (a B-pillar's paint
/// between the doors lies over the doors' skin; the pillar's own nodes are inside it).
/// Contact holds only those, so paint that followed an inner node (a door's inner panel,
/// a hood's inner frame) could go where the skin does not, into another car whose skin it
/// never crossed. The fit is a sheet's ([`flexskin::bind_sheet`]): the nodes lie in one
/// surface and the paint's height over it is the offset's
/// (`engine/docs/generated-vehicles-lessons.md`, "Paint binds to the collision skin").
pub fn binds_to_skin(glb: &Glb, m: &crate::glb::Mesh, rigid: &dyn Fn(&str) -> bool) -> bool {
    !rigid(&m.name)
        && !m.name.starts_with("glass_")
        && m.prims.iter().any(|p| glb.materials.get(p.material).is_some_and(|x| x.name == "paint"))
}

/// The outer collision triangles in a grid, to find the skin under a paint vertex.
struct SkinUnder {
    tris: Vec<[usize; 3]>,
    grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
}

/// The grid's cell (m), and how far from a paint vertex its skin is looked for: past the
/// paint's bulge over the triangles between the nodes (`CONTACT_OFFSET_CAP`, 0.12 m).
const UNDER_CELL: f32 = 0.1;
const UNDER_REACH: f32 = 0.15;

impl SkinUnder {
    fn new(car: &Car) -> SkinUnder {
        let tris: Vec<[usize; 3]> = car.tris.iter().filter(|t| !t.inner).map(|t| t.n).collect();
        let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> = Default::default();
        for (k, t) in tris.iter().enumerate() {
            let [a, b, c] = t.map(|i| car.nodes[i].p);
            let (lo, hi) = (cell(a.min(b).min(c)), cell(a.max(b).max(c)));
            for x in lo.0..=hi.0 {
                for y in lo.1..=hi.1 {
                    for z in lo.2..=hi.2 {
                        grid.entry((x, y, z)).or_default().push(k);
                    }
                }
            }
        }
        SkinUnder { tris, grid }
    }

    /// The nodes of the outer collision triangle nearest each of `verts` (within
    /// [`UNDER_REACH`]) that a paint mesh may follow: a part's structural nodes, as
    /// [`pool_with`] takes them, on any part. In first-found order (deterministic).
    fn nodes_under(&self, car: &Car, verts: &[Vec3]) -> Vec<u32> {
        let usable = |i: usize| {
            let n = &car.nodes[i];
            !n.contact && !n.no_group && n.sub.is_none() && !n.name.starts_with("hub")
        };
        let mut seen = vec![false; car.nodes.len()];
        let mut out = Vec::new();
        let r = (UNDER_REACH / UNDER_CELL).ceil() as i32;
        for &p in verts {
            let c = cell(p);
            let mut best = (UNDER_REACH, usize::MAX);
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        for &k in self.grid.get(&(c.0 + dx, c.1 + dy, c.2 + dz)).map_or(&[][..], |l| l.as_slice()) {
                            let [a, b, q] = self.tris[k].map(|i| car.nodes[i].p);
                            let d = crate::authoring::lint::tri_dist_pub(p, a, b, q);
                            if d < best.0 || (d == best.0 && k < best.1) {
                                best = (d, k);
                            }
                        }
                    }
                }
            }
            if best.1 == usize::MAX {
                continue;
            }
            for i in self.tris[best.1] {
                if usable(i) && !seen[i] {
                    seen[i] = true;
                    out.push(i as u32);
                }
            }
        }
        out
    }
}

fn cell(p: Vec3) -> (i32, i32, i32) {
    (
        (p.x / UNDER_CELL).floor() as i32,
        (p.y / UNDER_CELL).floor() as i32,
        (p.z / UNDER_CELL).floor() as i32,
    )
}

/// The skins of every mesh of `glb` over the rig's nodes (JBeam frame).
pub fn skins(glb: &Glb, car: &Car) -> Result<Skin, String> {
    skins_with(glb, car, super::surface::AXLE_F, &part_of, &rigid)
}

/// [`skins`] about a front axle at glTF z `axle_f`, meshes on parts by
/// `part_of`, rigid pieces by `rigid`.
pub fn skins_with(
    glb: &Glb,
    car: &Car,
    axle_f: f32,
    part_of: &dyn Fn(&str) -> Option<Part>,
    rigid: &dyn Fn(&str) -> bool,
) -> Result<Skin, String> {
    let to_jbeam = |g: Vec3| Vec3::new(g.x, axle_f - g.z, g.y);
    let nodes: Vec<String> = car.nodes.iter().map(|n| n.name.clone()).collect();
    let on_skin = outer_skin_nodes(car);
    let mut under: Option<SkinUnder> = None;
    let mut meshes = Vec::new();
    for m in &glb.meshes {
        let (mut vp, mut vn) = (Vec::new(), Vec::new());
        for p in &m.prims {
            vp.extend(p.pos.iter().map(|q| to_jbeam(Vec3::from_array(*q))));
            vn.extend(p.nrm.iter().map(|q| {
                let n = Vec3::from_array(*q);
                Vec3::new(n.x, -n.z, n.y)
            }));
        }
        let mut ids = pool_with(car, &m.name, part_of);
        // (RIG_PAINT_ANY_NODE=1: paint on every node of its part, as before.)
        let sheet = binds_to_skin(glb, m, rigid) && !std::env::var("RIG_PAINT_ANY_NODE").is_ok_and(|v| v == "1");
        if sheet {
            ids.retain(|&i| on_skin[i as usize]);
            // The skin the paint lies over where it is another part's (a B-pillar's paint
            // between the doors, whose own nodes are inside the doors' skin).
            let u = under.get_or_insert_with(|| SkinUnder::new(car));
            for i in u.nodes_under(car, &vp) {
                if !ids.contains(&i) {
                    ids.push(i);
                }
            }
        }
        if ids.len() < 3 {
            return Err(format!("{}: {} nodes to bind to", m.name, ids.len()));
        }
        let pos: Vec<Vec3> = ids.iter().map(|&i| car.nodes[i as usize].p).collect();
        let (verts, _) = if rigid(&m.name) {
            flexskin::bind_rigid(&ids, &pos, &vp, &vn)
        } else if m.name.starts_with("glass_") {
            flexskin::bind_with(&ids, &pos, &vp, &vn, Some(GLASS_AMP_MAX))
        } else if sheet {
            flexskin::bind_sheet(&ids, &pos, &vp, &vn)
        } else {
            flexskin::bind(&ids, &pos, &vp, &vn)
        }
        .map_err(|e| format!("{}: {e}", m.name))?;
        meshes.push(MeshSkin {
            name: m.name.clone(),
            verts,
        });
    }
    Ok(Skin { nodes, meshes })
}

/// Flexbody rows for the model's meshes (each on its part's group, its skin
/// in [`SKIN_FILE`], placed at the front axle: the mesh's origin is midway
/// between the axles).
pub fn flex_rows(glb: &Glb) -> Vec<(Part, Flex)> {
    flex_rows_with(glb, super::surface::AXLE_F, &part_of, SKIN_FILE)
}

/// [`flex_rows`] about a front axle at glTF z `axle_f`, meshes on parts by
/// `part_of`, skins in `skin_file`.
pub fn flex_rows_with(
    glb: &Glb,
    axle_f: f32,
    part_of: &dyn Fn(&str) -> Option<Part>,
    skin_file: &str,
) -> Vec<(Part, Flex)> {
    glb.meshes
        .iter()
        .map(|m| {
            let part = part_of(&m.name).unwrap_or(Part::Body);
            let deform = deform_group(&m.name);
            (
                part,
                Flex {
                    mesh: m.name.clone(),
                    groups: vec![part.name().to_string()],
                    deform_group: deform,
                    skin: Some(skin_file.into()),
                    pos: Some([0.0, axle_f, 0.0]),
                    owner: None,
                },
            )
        })
        .collect()
}

fn deform_group(mesh: &str) -> Option<String> {
    if mesh.starts_with("glass_") {
        return Some(mesh.to_string());
    }
    if mesh.starts_with("headlight_") || mesh.starts_with("taillight_") {
        let side = mesh.split('_').nth(1).unwrap_or("L");
        let kind = mesh.split('_').next().unwrap_or("");
        return Some(format!("{kind}_{side}"));
    }
    None
}

/// The deform groups (glass and lamps): the nodes of the mesh's part nearest
/// it carry it; it breaks once their beams yield (ratio).
pub fn deform_groups(glb: &Glb, car: &Car) -> Vec<(String, Vec<usize>, f32)> {
    deform_groups_with(glb, car, super::surface::AXLE_F, &part_of)
}

/// [`deform_groups`] about a front axle at glTF z `axle_f`, meshes on parts
/// by `part_of`.
pub fn deform_groups_with(
    glb: &Glb,
    car: &Car,
    axle_f: f32,
    part_of: &dyn Fn(&str) -> Option<Part>,
) -> Vec<(String, Vec<usize>, f32)> {
    let to_jbeam = |g: Vec3| Vec3::new(g.x, axle_f - g.z, g.y);
    let mut out: Vec<(String, Vec<usize>, f32)> = Vec::new();
    for m in &glb.meshes {
        let Some(g) = deform_group(&m.name) else {
            continue;
        };
        if out.iter().any(|o| o.0 == g) {
            continue;
        }
        let pts: Vec<Vec3> = m
            .prims
            .iter()
            .flat_map(|p| p.pos.iter().map(|q| to_jbeam(Vec3::from_array(*q))))
            .collect();
        let ids = pool_with(car, &m.name, part_of);
        let near: Vec<usize> = ids
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| {
                let p = car.nodes[i].p;
                pts.iter().any(|q| q.distance(p) < 0.15)
            })
            .collect();
        let ratio = if g.starts_with("glass") { 0.02 } else { 0.03 };
        out.push((g, near, ratio));
    }
    out
}

/// The rig, the model's glb and its skins.
pub struct Gen3 {
    pub rig: Rig,
    pub glb: Glb,
    pub skin: Skin,
}

pub fn build() -> Gen3 {
    let model = super::model::build();
    let glb = super::model::to_glb(&model);
    let rig = rig::rig(&model);
    let skin = skins(&glb, &rig.car).expect("gen3 skins");
    skin.packable().expect("gen3 skins fit the sidecar");
    Gen3 { rig, glb, skin }
}
