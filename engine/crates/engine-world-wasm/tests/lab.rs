//! The fracture lab's threshold mode on World draws each chunk with the lab scene's mesh
//! of the same index: the stress import's first chunks must be the asset's chunks (the lab
//! scene's bodies), in order, for every source the lab loads (`lab.rs` refuses an asset
//! where they are not).
use rubble::*;

fn glb(bytes: &[u8]) {
    let parsed = engine_fracture::parse_fracture_glb(bytes).unwrap();
    for level in engine_fracture::samples::initial_level(&parsed, None) {
        let desc = DestructibleAssetDesc::from_glb(bytes, level).unwrap();
        engine_world_wasm::check_lab_chunk_order(&desc).unwrap();
    }
}

#[test]
fn the_veneer_house_and_the_arch_keep_their_chunk_order() {
    glb(include_bytes!(
        "../../engine-webgpu/fracture/assets/veneer-house.glb"
    ));
    glb(include_bytes!(
        "../../engine-webgpu/fracture/assets/masonry-arch.glb"
    ));
}

#[test]
fn the_cuboid_house_keeps_its_chunk_order() {
    let desc = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    engine_world_wasm::check_lab_chunk_order(&desc).unwrap();
}

#[test]
fn scene_packs_keep_their_chunk_order() {
    let bytes = include_bytes!("../../engine-webgpu/fracture/packs/crush-wall.json");
    let desc = DestructibleAssetDesc::from_pack(bytes).unwrap();
    engine_world_wasm::check_lab_chunk_order(&desc).unwrap();
}

#[test]
fn a_shifted_numbering_is_refused() {
    let mut desc = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    desc.chunks.rotate_left(1);
    assert!(engine_world_wasm::check_lab_chunk_order(&desc).is_err());
}
