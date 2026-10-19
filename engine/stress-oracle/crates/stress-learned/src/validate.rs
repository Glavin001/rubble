//! Validate an on-disk `learned-impact/1` event.

use std::path::Path;

use crate::catalog::{self, Catalog};
use crate::constants::{CHUNK_STATE_DIM, JOINT_STATE_DIM, K, LOCAL6_DIM, SCHEMA_VERSION};
use crate::recorder::Manifest;

#[derive(Debug)]
pub struct ValidationError(pub String);

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ValidationError {}

/// Load manifest + catalogs and check shapes, endpoints, fail-class range.
pub fn validate_event(root: &Path) -> Result<Manifest, ValidationError> {
    let err = |s: String| ValidationError(s);
    let text = std::fs::read_to_string(root.join("manifest.json")).map_err(|e| err(e.to_string()))?;
    let man: Manifest = serde_json::from_str(&text).map_err(|e| err(e.to_string()))?;
    if man.schema_version != SCHEMA_VERSION {
        return Err(err(format!("schema_version {} != {SCHEMA_VERSION}", man.schema_version)));
    }
    if man.k as usize != K {
        return Err(err(format!("K={} != {K}", man.k)));
    }
    let n = man.n_chunks as usize;
    let e = man.n_bonds as usize;
    if man.n_frames == 0 {
        return Err(err("n_frames == 0".into()));
    }

    let static_cat = Catalog::load(&root.join("catalog.json")).map_err(err)?;
    require_shape(&static_cat, "chunk_mass", "f32", &[n])?;
    require_shape(&static_cat, "chunk_inertia", "f32", &[n, 3])?;
    require_shape(&static_cat, "bond_endpoints", "i32", &[e, 2])?;
    require_shape(&static_cat, "bond_area", "f32", &[e])?;
    // Spot-check a few more required static names.
    for name in [
        "chunk_half_extents",
        "chunk_material_id",
        "chunk_support",
        "chunk_level",
        "chunk_coarse_parent",
        "bond_width",
        "bond_offset",
        "bond_stiffness",
        "bond_strength",
        "bond_fracture_energy",
        "bond_rebar",
        "bond_weibull",
        "bond_buckling_length",
    ] {
        if !static_cat.arrays.contains_key(name) {
            return Err(err(format!("missing static array {name}")));
        }
    }
    let endpoints = catalog::read_i32(root, static_cat.arrays.get("bond_endpoints").unwrap()).map_err(err)?;
    for (bi, pair) in endpoints.chunks_exact(2).enumerate() {
        let (i, j) = (pair[0], pair[1]);
        if i < 0 || j < 0 || i as usize >= n || j as usize >= n || i == j {
            return Err(err(format!("bond {bi} endpoints [{i},{j}] invalid for N={n}")));
        }
    }

    for fi in 0..man.n_frames {
        let dir = catalog::frame_dir(root, fi);
        let cat = Catalog::load(&dir.join("catalog.json")).map_err(err)?;
        require_shape(&cat, "chunk_state", "f32", &[n, CHUNK_STATE_DIM])?;
        require_shape(&cat, "chunk_load_bins", "f32", &[n, K, LOCAL6_DIM])?;
        require_shape(&cat, "bond_joint_state", "f32", &[e, JOINT_STATE_DIM])?;
        require_shape(&cat, "bond_qs_force", "f32", &[e, LOCAL6_DIM])?;
        require_shape(&cat, "bond_fail_class", "i32", &[e])?;
        require_shape(&cat, "bond_peak_util", "f32", &[e])?;
        require_shape(&cat, "chunk_state_delta", "f32", &[n, CHUNK_STATE_DIM])?;
        let classes = catalog::read_i32(&dir, cat.arrays.get("bond_fail_class").unwrap()).map_err(err)?;
        for (i, &c) in classes.iter().enumerate() {
            if c < 0 || c as usize > K {
                return Err(err(format!("frame {fi} bond_fail_class[{i}]={c}")));
            }
        }
        // Ensure binaries exist and match byte length.
        for (name, entry) in &cat.arrays {
            let path = dir.join(&entry.path);
            let meta = std::fs::metadata(&path).map_err(|e| err(format!("{name}: {e}")))?;
            let want = entry.shape.iter().product::<usize>() * 4;
            if meta.len() as usize != want {
                return Err(err(format!(
                    "frame {fi} {name}: {} bytes, expected {want}",
                    meta.len()
                )));
            }
        }
    }
    Ok(man)
}

fn require_shape(cat: &Catalog, name: &str, dtype: &str, shape: &[usize]) -> Result<(), ValidationError> {
    let e = cat.arrays.get(name).ok_or_else(|| ValidationError(format!("missing {name}")))?;
    if e.dtype != dtype {
        return Err(ValidationError(format!("{name}: dtype {} != {dtype}", e.dtype)));
    }
    if e.shape != shape {
        return Err(ValidationError(format!("{name}: shape {:?} != {:?}", e.shape, shape)));
    }
    Ok(())
}
