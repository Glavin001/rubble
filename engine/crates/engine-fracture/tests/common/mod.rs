#![allow(dead_code)]

use engine_fracture::{parse_fracture_glb, FractureAsset};
use std::path::PathBuf;
use std::sync::OnceLock;

pub fn asset_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/fracture/assets")
        .join(file)
}

pub fn load(file: &str) -> FractureAsset {
    parse_fracture_glb(&std::fs::read(asset_path(file)).unwrap()).unwrap()
}

pub fn house() -> &'static FractureAsset {
    static HOUSE: OnceLock<FractureAsset> = OnceLock::new();
    HOUSE.get_or_init(|| load("veneer-house.glb"))
}

pub fn arch() -> &'static FractureAsset {
    static ARCH: OnceLock<FractureAsset> = OnceLock::new();
    ARCH.get_or_init(|| load("masonry-arch.glb"))
}

pub fn close(a: f64, b: f64, rel: f64) {
    assert!(
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-30),
        "{a} vs {b}"
    );
}
