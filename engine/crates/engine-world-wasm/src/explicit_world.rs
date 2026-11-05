//! Explicit soft-body vehicles in the World (`explicit-world.html`): the crash town
//! (`rubble::explicit_scene::crash_town`, the native benchmark's builder) on the page's
//! device, and the buffers a renderer reads in place (no readback).

use crate::{err, gpu_buffer, res, WebWorld};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use rubble::explicit_scene::{crash_town, town_coupling, town_layout};
use rubble::GpuContext;
use wasm_bindgen::prelude::*;

fn num(v: &serde_json::Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

/// `{ houses, ground, centres: [[x, y, z], ...] }` of a crash town for `count` vehicles
/// (at least one house per four).
#[wasm_bindgen(js_name = vehicleTownLayout)]
pub fn vehicle_town_layout(houses: u32, count: u32) -> Result<JsValue, JsValue> {
    let (centres, ground) = res(town_layout(houses as usize, count as usize))?;
    Ok(js(serde_json::json!({ "houses": centres.len(), "ground": ground, "centres": centres })))
}

#[wasm_bindgen]
impl WebWorld {
    /// The crash town on a new device. `spec`: `{ truck: <.truck text>, count, houses,
    /// speed (m/s), runUp (m), drag (default true) }`: `count` copies of the vehicle into
    /// prefractured cuboid houses (threshold destructibles), one from each side of each,
    /// past a crate each.
    #[wasm_bindgen(js_name = createVehicleTown)]
    pub async fn create_vehicle_town(spec: JsValue) -> Result<WebWorld, JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        let truck = spec
            .get("truck")
            .and_then(|t| t.as_str())
            .ok_or_else(|| err("createVehicleTown: spec.truck (.truck text)"))?;
        let drag = spec.get("drag").and_then(|x| x.as_bool()).unwrap_or(true);
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            drag,
            turbulence: drag,
            ..Default::default()
        };
        let template = ExplicitBeams::from_truck(truck, glam::Vec3::ZERO, glam::Vec3::ZERO, settings).map_err(err)?;
        let (info, device, queue) = crate::web_device().await?;
        let town = res(crash_town(
            GpuContext::Shared { info, device, queue },
            &template,
            num(&spec, "houses", 0.0) as usize,
            num(&spec, "count", 4.0) as usize,
            num(&spec, "speed", 15.0) as f32,
            num(&spec, "runUp", 8.0) as f32,
            0,
            &town_coupling(),
        ))?;
        crate::creation_check(&town.world).await?;
        Ok(WebWorld { world: town.world })
    }

    /// The explicit vehicles' node state `GPUBuffer` (3 `vec4<f32>` per node: position
    /// relative to its vehicle's origin and mass, velocity, force).
    #[wasm_bindgen(js_name = explicitNodesBuffer)]
    pub fn explicit_nodes_buffer(&self) -> JsValue {
        self.world.explicit_vehicles().map_or(JsValue::NULL, |g| gpu_buffer(g.nodes_buffer()))
    }

    /// Per vehicle `vec4<u32>`: its origin's bits (xyz).
    #[wasm_bindgen(js_name = explicitMiscBuffer)]
    pub fn explicit_misc_buffer(&self) -> JsValue {
        self.world.explicit_vehicles().map_or(JsValue::NULL, |g| gpu_buffer(g.misc_buffer()))
    }

    /// Collision triangles: `vec4<u32>` per triangle (global nodes a, b, c, vehicle).
    #[wasm_bindgen(js_name = explicitTrianglesBuffer)]
    pub fn explicit_triangles_buffer(&self) -> JsValue {
        self.world.explicit_triangles().map_or(JsValue::NULL, |(b, _)| gpu_buffer(b))
    }

    #[wasm_bindgen(js_name = explicitTriangleCount)]
    pub fn explicit_triangle_count(&self) -> u32 {
        self.world.explicit_triangles().map_or(0, |(_, n)| n)
    }

    /// `[explicit steps, World steps]` the vehicles have run.
    #[wasm_bindgen(js_name = explicitSteps)]
    pub fn explicit_steps(&self) -> Vec<f64> {
        self.world
            .explicit_vehicle_steps()
            .map_or(vec![0.0, 0.0], |(s, w)| vec![s as f64, w as f64])
    }
}
