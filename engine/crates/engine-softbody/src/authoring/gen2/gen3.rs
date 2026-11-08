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
    match part_of(mesh) {
        Some(p) => (0..car.nodes.len() as u32)
            .filter(|&i| {
                let n = &car.nodes[i as usize];
                n.part == p && !n.contact && !n.no_group && !n.name.starts_with("hub")
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
    let mut meshes = Vec::new();
    for m in &glb.meshes {
        let ids = pool_with(car, &m.name, part_of);
        if ids.len() < 3 {
            return Err(format!("{}: {} nodes to bind to", m.name, ids.len()));
        }
        let pos: Vec<Vec3> = ids.iter().map(|&i| car.nodes[i as usize].p).collect();
        let (mut vp, mut vn) = (Vec::new(), Vec::new());
        for p in &m.prims {
            vp.extend(p.pos.iter().map(|q| to_jbeam(Vec3::from_array(*q))));
            vn.extend(p.nrm.iter().map(|q| {
                let n = Vec3::from_array(*q);
                Vec3::new(n.x, -n.z, n.y)
            }));
        }
        let (verts, _) = if rigid(&m.name) {
            flexskin::bind_rigid(&ids, &pos, &vp, &vn)
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
