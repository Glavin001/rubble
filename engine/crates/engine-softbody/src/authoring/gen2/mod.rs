//! `gen2`: Rubble's procedural vehicle generator that writes BeamNG content.
//!
//! One Rust source builds the physics skeleton and the render mesh together
//! from one analytic shape ([`shape`]): the node–beam car ([`sedan`], with the
//! builder's mass, stiffness and yield rules in [`build`]) is written as JBeam
//! parts and a `.pc` ([`export`]), and the visual as a Collada mesh with its
//! `materials.json` whose meshes bind to their part's node group as flexbodies
//! ([`mesh`]). The generated car then loads through the same JBeam loader as
//! community content (CPU, GPU, the page's `?mod=`). Guide:
//! `engine/docs/vehicle-gen2.md`.

pub mod build;
pub mod export;
pub mod gen3;
pub mod mesh;
pub mod model;
pub mod rig;
pub mod sedan;
pub mod shape;
pub mod surface;
pub mod vg_input;

pub use build::{Car, Part, Role};

/// Every file of the generated sedan (gen3: the render model rigged), paths
/// relative to a mod folder (the folder the page's `?mod=` points at): the
/// JBeam parts and `.pc` under `gen2/`, the render model `gen2/gen2.glb` and
/// its skins `gen2/gen2.rskin` (the flexbody rows name both), `sedan.glb` (the
/// static model for `showcase-car.html?glb=`), then `files.json`.
pub fn files() -> Vec<(String, Vec<u8>)> {
    files_of(&gen3::build())
}

/// [`files`] of a built gen3 car.
pub fn files_of(g: &gen3::Gen3) -> Vec<(String, Vec<u8>)> {
    let flex = gen3::flex_rows(&g.glb);
    let groups = gen3::deform_groups(&g.glb, &g.rig.car);
    let mut out: Vec<(String, Vec<u8>)> = export::jbeam_files(&g.rig.car, &flex, &groups)
        .into_iter()
        .map(|(p, t)| (p, t.into_bytes()))
        .collect();
    let glb = crate::glb::write(&g.glb);
    out.push((format!("{}/{}", export::NAME, gen3::GLB_FILE), glb.clone()));
    out.push((format!("{}/{}", export::NAME, gen3::SKIN_FILE), g.skin.encode()));
    out.push(("sedan.glb".into(), glb));
    let list: Vec<&str> = out.iter().map(|(p, _)| p.as_str()).collect();
    let index = serde_json::to_string(&list).unwrap();
    out.push(("files.json".into(), index.into_bytes()));
    out
}

/// The first (gen2) lattice's files, kept for comparison: its own Collada
/// mesh on nearest-node frames.
pub fn files_gen2() -> Vec<(String, Vec<u8>)> {
    let car = sedan::sedan();
    let vis = mesh::build(&car);
    let mut out: Vec<(String, Vec<u8>)> = export::jbeam_files(&car, &vis.flex, &vis.deform_groups)
        .into_iter()
        .map(|(p, t)| (p, t.into_bytes()))
        .collect();
    out.push((format!("{}/{}.dae", export::NAME, export::NAME), vis.collada.into_bytes()));
    out.push((format!("{}/{}.materials.json", export::NAME, export::NAME), vis.materials.into_bytes()));
    let list: Vec<&str> = out.iter().map(|(p, _)| p.as_str()).collect();
    let index = serde_json::to_string(&list).unwrap();
    out.push(("files.json".into(), index.into_bytes()));
    out
}
