//! The one-World showcase (`one-world.html`): `rubble::showcase_scene`'s staged crashes
//! on the page's device, the layout a page frames them by, and the explicit vehicles'
//! buffers a skinned renderer (`showcase-car-render.mjs`) reads in place.

use crate::{err, gpu_buffer, res, WebWorld};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use rubble::explicit_scene::town_coupling;
use rubble::showcase_scene::{
    showcase, showcase_layout, ShowcaseHouses, ShowcaseLayout, ShowcasePreset,
};
use rubble::GpuContext;
use wasm_bindgen::prelude::*;

fn num(v: &serde_json::Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

fn preset(v: &serde_json::Value) -> Result<ShowcasePreset, JsValue> {
    let name = v.get("preset").and_then(|p| p.as_str()).unwrap_or("house");
    ShowcasePreset::parse(name).ok_or_else(|| {
        err(format!(
            "unknown preset `{name}`: house, crates, twoway, convoy"
        ))
    })
}

fn layout_js(l: &ShowcaseLayout) -> JsValue {
    use serde::Serialize;
    serde_json::json!({
        "preset": l.preset,
        "ground": l.ground,
        "houses": l.houses,
        "vehicles": l.vehicles.iter().map(|v| serde_json::json!({ "front": v.front, "speed": v.speed, "turns": v.turns })).collect::<Vec<_>>(),
        "impact": l.impact,
        "impactTime": l.impact_time,
        "ropes": l.ropes.iter().map(|r| serde_json::json!({ "anchor": r.anchor, "body": r.body, "radius": r.radius })).collect::<Vec<_>>(),
        "chunks": l.chunks,
        "crates": l.crates,
    })
    .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    .unwrap_or(JsValue::NULL)
}

/// `{ preset, ground, houses: [[x, y, z]], vehicles: [{ front, speed, turns }], impact,
/// impactTime (s), ropes: [{ anchor, body, radius }], chunks, crates }` of a showcase
/// preset (`spec` as [`WebWorld::create_showcase`]'s, without the truck).
#[wasm_bindgen(js_name = showcaseLayout)]
pub fn showcase_layout_js(spec: JsValue) -> Result<JsValue, JsValue> {
    let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
    let l = res(showcase_layout(
        preset(&spec)?,
        num(&spec, "count", 1.0) as usize,
        num(&spec, "speed", 0.0) as f32,
        num(&spec, "runUp", 0.0) as f32,
    ))?;
    Ok(layout_js(&l))
}

#[wasm_bindgen]
impl WebWorld {
    /// A showcase preset on a new device. `spec`: `{ truck: <.truck text>, preset:
    /// 'house' | 'crates' | 'twoway' | 'convoy', count (convoy: 2 to 4), speed (m/s, 0:
    /// the preset's), runUp (m, 0: the preset's), drag (default true), strength (× every
    /// house bond's capacity) and supportLoss (default: `ShowcaseHouses::showcase()`) }`.
    #[wasm_bindgen(js_name = createShowcase)]
    pub async fn create_showcase(spec: JsValue) -> Result<WebWorld, JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        let truck = spec
            .get("truck")
            .and_then(|t| t.as_str())
            .ok_or_else(|| err("createShowcase: spec.truck (.truck text)"))?;
        let drag = spec.get("drag").and_then(|x| x.as_bool()).unwrap_or(true);
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            drag,
            turbulence: drag,
            ..Default::default()
        };
        let template =
            ExplicitBeams::from_truck(truck, glam::Vec3::ZERO, glam::Vec3::ZERO, settings)
                .map_err(err)?;
        let p = preset(&spec)?;
        let count = num(&spec, "count", 1.0) as usize;
        let speed = num(&spec, "speed", 0.0) as f32;
        let run_up = num(&spec, "runUp", 0.0) as f32;
        let n = res(showcase_layout(p, count, speed, run_up))?
            .vehicles
            .len();
        let (info, device, queue) = crate::web_device().await?;
        // The vehicles' kernels compiled without blocking first (as `createVehicleTown`).
        let fleet = vec![template.clone(); n];
        let specs =
            engine_webgpu::ExplicitBeamsGpu::compute_specs(&fleet, Some(town_coupling().capacity));
        crate::warm_pipelines(&device, &specs, std::rc::Rc::new(|_: &str, _: f64| {})).await;
        let s = res(showcase(
            GpuContext::Shared {
                info,
                device,
                queue,
            },
            &template,
            p,
            count,
            speed,
            run_up,
            ShowcaseHouses {
                strength_scale: num(
                    &spec,
                    "strength",
                    ShowcaseHouses::showcase().strength_scale as f64,
                ) as f32,
                support_loss: spec
                    .get("supportLoss")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(ShowcaseHouses::showcase().support_loss),
            },
            &town_coupling(),
        ))?;
        crate::creation_check(&s.world).await?;
        Ok(WebWorld { world: s.world })
    }

    /// The explicit vehicles' beam state `GPUBuffer` (2 `vec4<f32>` per beam: rest
    /// length, yields, gate; strength, last force, broken 0 or 1).
    #[wasm_bindgen(js_name = explicitBeamsBuffer)]
    pub fn explicit_beams_buffer(&self) -> JsValue {
        self.world
            .explicit_vehicles()
            .map_or(JsValue::NULL, |g| gpu_buffer(g.beams_buffer()))
    }

    /// Per vehicle `[first node, first beam]` in the node and beam buffers.
    #[wasm_bindgen(js_name = explicitSpans)]
    pub fn explicit_spans(&self) -> Vec<u32> {
        self.world.explicit_vehicles().map_or(Vec::new(), |g| {
            (0..g.vehicle_count())
                .flat_map(|v| [g.node_offset(v), g.beam_offset(v)])
                .collect()
        })
    }
}
