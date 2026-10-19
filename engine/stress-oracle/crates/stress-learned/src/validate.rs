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

    let context_path = root.join("context.json");
    let preview = if context_path.exists() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(context_path).map_err(|e| err(e.to_string()))?).map_err(|e| err(e.to_string()))?;
        if let Some(history)=v.get("joint_history") {
            if history["format"]!=crate::joint_history::FORMAT || history["path"]!="joint_history/manifest.json" {
                return Err(err("unknown joint-history contract".into()));
            }
            let keys:Vec<(usize,usize)>=serde_json::from_value(v["bond_keys"].clone()).map_err(|e|err(e.to_string()))?;
            if keys.len()!=e || keys.iter().collect::<std::collections::HashSet<_>>().len()!=e {return Err(err("joint-history bond key mismatch".into()));}
            crate::joint_history::validate(root,&keys,man.n_frames as usize,man.dt).map_err(err)?;
        }
        if let Some(response)=v.get("bond_response") {
            if response["format"]!=crate::response_history::FORMAT || response["path"]!="bond_response/manifest.json" || v.get("joint_history").is_none() {
                return Err(err("unknown or incomplete bond-response contract".into()));
            }
            let keys:Vec<(usize,usize)>=serde_json::from_value(v["bond_keys"].clone()).map_err(|e|err(e.to_string()))?;
            crate::response_history::validate(root,&keys,man.n_frames as usize,man.dt).map_err(err)?;
        }
        match v.get("load_preview").and_then(|p| p.get("format")).and_then(|f| f.as_str()) {
            None => false,
            Some("rigid-contact-preview/1") => true,
            Some(f) => return Err(err(format!("unknown load preview format {f}"))),
        }
    } else { false };
    for fi in 0..man.n_frames {
        let dir = catalog::frame_dir(root, fi);
        let cat = Catalog::load(&dir.join("catalog.json")).map_err(err)?;
        if preview {
            require_shape(&cat, "chunk_preview_load_bins", "f32", &[n, K, LOCAL6_DIM])?;
            require_shape(&cat, "chunk_preview_contact", "i32", &[n])?;
            let loads = catalog::read_f32(&dir, &cat.arrays["chunk_preview_load_bins"]).map_err(err)?;
            let contact = catalog::read_i32(&dir, &cat.arrays["chunk_preview_contact"]).map_err(err)?;
            if loads.iter().any(|x| !x.is_finite()) || contact.iter().any(|&x| x != 0 && x != 1) {
                return Err(err(format!("invalid rigid preview in frame {fi}")));
            }
        } else if cat.arrays.contains_key("chunk_preview_load_bins") || cat.arrays.contains_key("chunk_preview_contact") {
            return Err(err("preview arrays without context contract".into()));
        }
        require_shape(&cat, "chunk_state", "f32", &[n, CHUNK_STATE_DIM])?;
        require_shape(&cat, "chunk_load_bins", "f32", &[n, K, LOCAL6_DIM])?;
        require_shape(&cat, "chunk_contact", "i32", &[n])?;
        require_shape(&cat, "chunk_gravity_local", "f32", &[n, 3])?;
        require_shape(&cat, "bond_joint_state", "f32", &[e, JOINT_STATE_DIM])?;
        require_shape(&cat, "bond_force_prev", "f32", &[e, LOCAL6_DIM])?;
        require_shape(&cat, "bond_qs_force", "f32", &[e, LOCAL6_DIM])?;
        require_shape(&cat, "bond_fail_class", "i32", &[e])?;
        require_shape(&cat, "bond_peak_util", "f32", &[e])?;
        require_shape(&cat, "bond_fail_mode", "i32", &[e])?;
        require_shape(&cat, "bond_joint_state_end", "f32", &[e, JOINT_STATE_DIM])?;
        require_shape(&cat, "bond_force_end", "f32", &[e, LOCAL6_DIM])?;
        require_shape(&cat, "chunk_state_delta", "f32", &[n, CHUNK_STATE_DIM])?;
        let classes = catalog::read_i32(&dir, cat.arrays.get("bond_fail_class").unwrap()).map_err(err)?;
        for (i, &c) in classes.iter().enumerate() {
            if c < 0 || c as usize > K {
                return Err(err(format!("frame {fi} bond_fail_class[{i}]={c}")));
            }
        }
        let modes = catalog::read_i32(&dir, cat.arrays.get("bond_fail_mode").unwrap()).map_err(err)?;
        for (i, &m) in modes.iter().enumerate() {
            // SCHEMA: 0 none, 1 tension, 2 shear, 3 compression, 4 buckling, 5 rebar
            if m < 0 || m > 5 {
                return Err(err(format!("frame {fi} bond_fail_mode[{i}]={m}")));
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
