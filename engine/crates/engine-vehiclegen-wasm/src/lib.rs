//! wasm-bindgen face of `engine-vehiclegen`: build a vehicle from parameters
//! in the browser. The page (engine-webgpu `vehicle-designer.html`) runs it in
//! a worker and draws the returned buffers; all geometry is made here, in
//! Rust, from the same code as the `vehiclegen` CLI.

use engine_vehiclegen::export::{glb, render_parts, RenderPart};
use engine_vehiclegen::{build, design_from_json, BuildOpts, Built};
use js_sys::{Float32Array, Uint32Array};
use wasm_bindgen::prelude::*;

/// A built vehicle as render buffers, part by part.
#[wasm_bindgen]
pub struct VehicleBuild {
    parts: Vec<RenderPart>,
    built: Built,
}

#[wasm_bindgen]
impl VehicleBuild {
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }
    pub fn part_name(&self, i: usize) -> String {
        self.parts[i].name.clone()
    }
    pub fn positions(&self, i: usize) -> Float32Array {
        Float32Array::from(&self.parts[i].positions[..])
    }
    pub fn normals(&self, i: usize) -> Float32Array {
        Float32Array::from(&self.parts[i].normals[..])
    }
    pub fn uvs(&self, i: usize) -> Float32Array {
        Float32Array::from(&self.parts[i].uvs[..])
    }
    /// RGBA per vertex, or an empty array when the part has none.
    pub fn colors(&self, i: usize) -> Float32Array {
        match &self.parts[i].colors {
            Some(c) => Float32Array::from(&c[..]),
            None => Float32Array::new_with_length(0),
        }
    }
    /// Surface tag per vertex of a solid part: 0 outer skin, 1 edge wall,
    /// 2 inner face (all 0 for parts that are not skin + walls + inner face).
    pub fn surfaces(&self, i: usize) -> js_sys::Uint8Array {
        let s: Vec<u8> = self.parts[i].surf.iter().map(|s| *s as u8).collect();
        js_sys::Uint8Array::from(&s[..])
    }
    pub fn indices(&self, i: usize) -> Uint32Array {
        Uint32Array::from(&self.parts[i].indices[..])
    }
    /// [material, first index, index count, ...] (material indexes
    /// `material_names()`).
    pub fn groups(&self, i: usize) -> Uint32Array {
        let g: Vec<u32> = self.parts[i].groups.iter().flat_map(|&(m, s, c)| [m as u32, s, c]).collect();
        Uint32Array::from(&g[..])
    }
    pub fn triangle_count(&self) -> usize {
        self.parts.iter().map(|p| p.indices.len() / 3).sum()
    }
    /// Markers as JSON: [{name, pos: [x, y, z], xaxis: [x, y, z]}].
    pub fn markers_json(&self) -> String {
        let m: Vec<serde_json::Value> = self
            .built
            .markers
            .iter()
            .map(|m| serde_json::json!({ "name": m.name, "pos": [m.pos.x, m.pos.y, m.pos.z], "xaxis": [m.xaxis.x, m.xaxis.y, m.xaxis.z] }))
            .collect();
        serde_json::to_string(&m).unwrap()
    }
    /// Paint as [r, g, b, metallic, roughness] (linear RGB).
    pub fn paint(&self) -> Vec<f64> {
        let p = self.built.paint;
        vec![p[0], p[1], p[2], self.built.metallic, self.built.roughness]
    }
    /// The build as a glTF 2.0 binary (the art-spec delivery).
    pub fn to_glb(&self) -> Vec<u8> {
        glb(&self.parts, &self.built.markers, self.built.paint, self.built.metallic, self.built.roughness)
    }
}

/// Build `design` from JSON parameters (missing fields take the defaults).
/// `uv`: panel UVs (girth table + atlas); the live preview passes false.
#[wasm_bindgen]
pub fn build_vehicle(design: &str, params_json: &str, uv: bool) -> Result<VehicleBuild, JsError> {
    let d = design_from_json(design, params_json).map_err(|e| JsError::new(&e))?;
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(d.as_ref(), BuildOpts { uv, ..Default::default() })))
        .map_err(|_| JsError::new("the generator failed for these parameters"))?;
    Ok(VehicleBuild { parts: render_parts(&built.parts), built })
}

/// The art spec's ten material names, in material-index order.
#[wasm_bindgen]
pub fn material_names() -> Vec<String> {
    engine_vehiclegen::mesh::MATERIALS.iter().map(|s| s.to_string()).collect()
}

/// A design's default parameters as JSON.
#[wasm_bindgen]
pub fn default_params(design: &str) -> Result<String, JsError> {
    engine_vehiclegen::default_params_json(design).map_err(|e| JsError::new(&e))
}
