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
pub mod mesh;
pub mod model;
pub mod sedan;
pub mod shape;
pub mod surface;

pub use build::{Car, Part, Role};

/// Every file of the generated sedan, paths relative to a mod folder (the
/// folder the page's `?mod=` points at): `gen2/*.jbeam`, `gen2/gen2.pc`,
/// `gen2/gen2.dae`, `gen2/gen2.materials.json`, then `files.json`.
pub fn files() -> Vec<(String, Vec<u8>)> {
    let car = sedan::sedan();
    let vis = mesh::build(&car);
    let mut out: Vec<(String, Vec<u8>)> = export::jbeam_files(&car, &vis.flex, &vis.deform_groups)
        .into_iter()
        .map(|(p, t)| (p, t.into_bytes()))
        .collect();
    out.push((format!("{}/{}.dae", export::NAME, export::NAME), vis.collada.into_bytes()));
    out.push((format!("{}/{}.materials.json", export::NAME, export::NAME), vis.materials.into_bytes()));
    // The render model to the vehicle art spec (`sedan.glb`, glTF axes).
    out.push(("sedan.glb".into(), crate::glb::write(&model::to_glb(&model::build()))));
    let list: Vec<&str> = out.iter().map(|(p, _)| p.as_str()).collect();
    let index = serde_json::to_string(&list).unwrap();
    out.push(("files.json".into(), index.into_bytes()));
    out
}
