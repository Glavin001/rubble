//! Directed debris (`World::set_guidance`, `set_guidance_zones`,
//! `body_set_guided_tags`) and non-blocking reads of its state.

use crate::{bits, err, res, WebWorld};
use engine_schema::guidance::{GuideBodyGpu, GuideEnd, ZoneDesc, GUIDE_SCRATCH_CLAIMS};
use rubble::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WebWorld {
    /// Enable, disable or retune guidance: `GuidanceSettings` (camelCase serde
    /// JSON, missing fields default).
    #[wasm_bindgen(js_name = setGuidance)]
    pub fn set_guidance(&mut self, settings: JsValue) -> Result<(), JsValue> {
        let v: serde_json::Value = serde_wasm_bindgen::from_value(settings).map_err(err)?;
        let s: GuidanceSettings = serde_json::from_value(v).map_err(|e| err(format!("GuidanceSettings: {e}")))?;
        res(self.world.set_guidance(&s))
    }

    /// Replace the zone list: `ZoneDesc` objects (camelCase serde JSON) whose
    /// `anchorBody` is a body slot (as `bodySlot` gives it), or absent.
    #[wasm_bindgen(js_name = setGuidanceZones)]
    pub fn set_guidance_zones(&mut self, zones: JsValue) -> Result<(), JsValue> {
        let v: serde_json::Value = serde_wasm_bindgen::from_value(zones).map_err(err)?;
        let zones: Vec<ZoneDesc> = serde_json::from_value(v).map_err(|e| err(format!("ZoneDesc: {e}")))?;
        let mut out = Vec::with_capacity(zones.len());
        for (k, z) in zones.into_iter().enumerate() {
            let anchor = match z.anchor_body {
                Some(i) => ZoneAnchor::Body(
                    self.world.body_at(i).ok_or_else(|| err(format!("zone {k}: no body in slot {i}")))?,
                ),
                None => ZoneAnchor::World(z.position),
            };
            out.push(GuidanceZone {
                anchor,
                anchor_relative: z.anchor_relative,
                yaw: z.yaw,
                offset: z.offset,
                shape: z.shape,
                weight: z.weight,
                quota: z.quota,
                active: z.active,
                tags: z.tags,
                motion: z.motion,
            });
        }
        res(self.world.set_guidance_zones(&out))
    }

    #[wasm_bindgen(js_name = resetGuidanceClaims)]
    pub fn reset_guidance_claims(&mut self) -> Result<(), JsValue> {
        res(self.world.reset_guidance_claims())
    }

    /// Guide a body toward zones whose tags meet `tagMask` (0: stop guiding it).
    #[wasm_bindgen(js_name = bodySetGuidedTags)]
    pub fn body_set_guided_tags(&mut self, body: f64, tag_mask: u32) -> Result<(), JsValue> {
        res(self.world.body_set_guided_tags(BodyId::from_bits(bits(body)), tag_mask))
    }

    /// Bodies per guidance state and end reason, resolved once read: `{ off,
    /// armed, steering, done, ended: { none, pushed, arrived, landed, short,
    /// slow, leftOut, noZone } }`, and each zone's claims (`claims`).
    #[wasm_bindgen(js_name = readGuidance)]
    pub fn read_guidance(&self) -> Result<js_sys::Promise, JsValue> {
        let (bodies, scratch, zones) = self
            .world
            .sim()
            .guidance_buffers()
            .ok_or_else(|| err("the world has no guidance (WorldDef.guidance)"))?;
        let stride = std::mem::size_of::<GuideBodyGpu>() as u64;
        let n = u64::from(self.world.sim().body_count());
        let ranges = vec![
            (bodies.clone(), 0, n * stride),
            (scratch.clone(), u64::from(GUIDE_SCRATCH_CLAIMS) * 4, u64::from(zones) * 4),
        ];
        let read = crate::read::read_ranges(&self.world, ranges);
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let words = js_sys::Uint32Array::from(wasm_bindgen_futures::JsFuture::from(read).await?).to_vec();
            let body_words = (n * stride / 4) as usize;
            let mut c = [0u32; 12];
            for r in bytemuck::cast_slice::<u32, GuideBodyGpu>(&words[..body_words]) {
                c[(r.state as usize).min(3)] += 1;
                c[4 + GuideEnd::of(r) as usize] += 1;
            }
            let names = ["none", "pushed", "arrived", "landed", "short", "slow", "leftOut", "noZone"];
            let ended: serde_json::Map<_, _> =
                names.iter().enumerate().map(|(k, n)| (n.to_string(), c[4 + k].into())).collect();
            let v = serde_json::json!({
                "off": c[0], "armed": c[1], "steering": c[2], "done": c[3], "ended": ended,
                "claims": &words[body_words..],
            });
            use serde::Serialize;
            v.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(err)
        }))
    }
}
